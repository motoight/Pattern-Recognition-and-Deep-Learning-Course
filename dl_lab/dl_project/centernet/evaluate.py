"""
author: Lei Wu, YueMing Wu
Notice that if there is any problem happen when execute this code
Maybe because there is something wrong with the path
Please contact us or refer to this link:
https://www.kaggle.com/motoight/fork-of-mycenternet
"""

import time
from tensorboardX import SummaryWriter
import torchvision
import torch
import lib.utils as utils
import nets
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import pandas as pd



DIR_TRAIN = 'data/images/train'
DIR_CSV = 'data/train.csv'
def process_bbox(df):
    df['bbox'] = df['bbox'].apply(lambda x: eval(x))
    df['x'] = df['bbox'].apply(lambda x: x[0])
    df['y'] = df['bbox'].apply(lambda x: x[1])
    df['w'] = df['bbox'].apply(lambda x: x[2])
    df['h'] = df['bbox'].apply(lambda x: x[3])
    df['x'] = df['x'].astype(np.float)
    df['y'] = df['y'].astype(np.float)
    df['w'] = df['w'].astype(np.float)
    df['h'] = df['h'].astype(np.float)

    df.drop(columns=['bbox'],inplace=True)
#     df.reset_index(drop=True)
    return df




df = pd.read_csv(DIR_CSV)
df_new = process_bbox(df)

# split train data into train and validation
image_ids = df_new['image_id'].unique()
train_ids = image_ids[0:int(0.8*len(image_ids))]
val_ids = image_ids[int(0.8*len(image_ids)):]
train_df = df_new[df_new['image_id'].isin(train_ids)]
val_df = df_new[df_new['image_id'].isin(val_ids)]

# get validate dataset
# set data augmentation to false
val_set = utils.Wheat(val_df,DIR_TRAIN,False)
# evaluate one image per step
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=1,
    shuffle=True,
    num_workers=0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = get_pose_net(num_layers=50, head_conv=64, num_classes=1)
# model.to(device)
model = utils.get_hourglass['small_hourglass']
model.to(device)
# state_path = '../input/0625chkpt/chkpt-resnet50fpn-0626-1.pt'
# state_path = "../input/0625chkpt/chkpt-resnet50-0612-1.pt"
state_path = "../input/0625chkpt/chkpt-smallhourglass-0627-1.pt"
state = torch.load(state_path)
for k in state:
    print(k)
model.load_state_dict(state['net'])


# calculate mAP@0.5
results1 = []
# calculate mAP@0.75
results2 = []

socre_bound = 0.2

t0 = time.perf_counter()

for sample in val_loader:
    for k in sample:
        if k != 'img_id' and k != 'boxes':
            sample[k] = sample[k].to(device)

    with torch.no_grad():
        img = sample['image']
        img_id = sample['img_id'][0]

        output = model(img)[-1]
        #         print('using: {}ms'.format(1000*dur))

        dets = utils.ctdet_decode(*output, K=50)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]
        dets[:, :2] = utils.transform_preds(dets[:, 0:2],
                                      sample['c'].cpu().numpy(),
                                      sample['s'].cpu().numpy(),
                                      (128, 128))
        dets[:, 2:4] = utils.transform_preds(dets[:, 2:4],
                                       sample['c'].cpu().numpy(),
                                       sample['s'].cpu().numpy(),
                                       (128, 128))
        #         dur =  time.perf_counter() - t0

        mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)
        boxes = dets[:, :4]
        scores = dets[:, -2]
        gt = sample['boxes'].numpy()[0]

        #         c_img = img.cpu().numpy()
        #         c_img = c_img[0].transpose(1,2,0)
        #         c_img = np.ascontiguousarray(c_img)


        boxes = np.clip(boxes, 0, 1024).astype(np.int32)
        precision1 = utils.calculate_image_precision(gt, boxes, (0.5,))
        results1.append(precision1)
        precision2 = utils.calculate_image_precision(gt, boxes, (0.75,))
        results2.append(precision2)

        if precision1 > 0.7:
            print("0.5mAP: {}".format(np.mean(precision1)))
            print("0.75mAP: {}".format(np.mean(precision2)))
            c_img = img.cpu().numpy()
            c_img = c_img[0].transpose(1, 2, 0)
            c_img = np.ascontiguousarray(c_img)
            c_img = (c_img * std) + mean
            c_img = cv2.resize(c_img, (1024, 1024))
            for i, box in enumerate(boxes):
                box = box.astype(np.int32)
                #                 if dets[i-1][-2]<0.2:
                #                     continue
                cv2.rectangle(c_img,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (200, 200, 200), 2)

            for i, box in enumerate(gt):
                box = box.astype(np.int32)
                #                 if dets[i-1][-2]<0.2:
                #                     continue
                cv2.rectangle(c_img,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (200, 0, 0), 2)

            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(111)
            ax1.imshow(c_img)

print("TOTAL 0.5mAP: {}".format(np.mean(results1)))
print("TOTAL 0.75mAP: {}".format(np.mean(results2)))