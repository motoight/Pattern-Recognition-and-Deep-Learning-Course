"""
author: Lei Wu, YueMing Wu
Notice that if there is any problem happen when execute this code
Maybe because there is something wrong with the path
Please contact us or refer to this link:
https://www.kaggle.com/motoight/mycenternet
"""

import time
from tensorboardX import SummaryWriter
import torchvision
import torch
import lib.utils as utils
import nets
import pandas as pd
import numpy as np

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
epoches = 70
lr = 0.005
log_interval = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_writer = SummaryWriter()
batch_size = 8


# img_ids = df['image_id'].unique()
# train_ids, val_ids = train_test_split(img_ids, test_size = 0.2,
#                                       shuffle=True,random_state=42)

# split train data into train and validation

image_ids = df_new['image_id'].unique()
train_ids = image_ids[0:int(0.8*len(image_ids))]
val_ids = image_ids[int(0.8*len(image_ids)):]
train_df = df_new[df_new['image_id'].isin(train_ids)]
val_df = df_new[df_new['image_id'].isin(val_ids)]

train_df = df[df['image_id'].isin(train_ids)]

train_set = utils.Wheat(train_df,DIR_TRAIN,True)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)


model = nets.get_pose_net_rpn(num_layers=50, head_conv=64, num_classes=1,verbose = False)
# model = nets.get_hourglass['small_hourglass']
# load_model(model, '../input/hourglass-chkpt/checkpoint.t7')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.5)

def train(epoch):
    print('\n Epoch: %d' % epoch)
    model.train()
    tic = time.perf_counter()

    t_hloss = 0
    t_whloss = 0
    t_regloss = 0

    for batch_idx, batch in enumerate(train_loader):
        for k in batch:
            if k != 'img_id':
                batch[k] = batch[k].to(device=device, non_blocking=True)
        outputs = model(batch['image'])
        hmap, regs, w_h_ = zip(*outputs)
        regs = [utils._tranpose_and_gather_feature(r, batch['inds']) for r in regs]
        w_h_ = [utils._tranpose_and_gather_feature(r, batch['inds']) for r in w_h_]

        hmap_loss = utils._neg_loss(hmap, batch['hmap'])
        reg_loss = utils._reg_loss(regs, batch['regs'], batch['ind_masks'])
        w_h_loss = utils._reg_loss(w_h_, batch['w_h_'], batch['ind_masks'])

        t_hloss += hmap_loss.item()
        t_whloss += w_h_loss.item()
        t_regloss += reg_loss.item()

        loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0 and batch_idx != 0:
            duration = time.perf_counter() - tic
            tic = time.perf_counter()
            #             print('[%d/%d-%d/%d] ' % (epoch, epoches, batch_idx*batch_size, len(train_loader.dataset)) +
            #                   ' hmap_loss= %.5f reg_loss= %.5f w_h_loss= %.5f' %
            #                   (hmap_loss.item(), reg_loss.item(), w_h_loss.item()) +
            #                   ' (%f samples/sec)' % (batch_size * log_interval / duration))
            print('[%d/%d-%d/%d] ' % (epoch, epoches, batch_idx * batch_size, len(train_loader.dataset)) +
                  ' hmap_loss= %.5f reg_loss= %.5f w_h_loss= %.5f' %
                  (t_hloss / log_interval, t_regloss / log_interval, t_whloss / log_interval) +
                  ' (%f samples/sec)' % (batch_size * log_interval / duration))

            t_hloss = 0
            t_whloss = 0
            t_regloss = 0

            step = len(train_loader) * epoch + batch_idx
            summary_writer.add_scalar('hmap_loss', hmap_loss.item(), step)
            summary_writer.add_scalar('reg_loss', reg_loss.item(), step)
            summary_writer.add_scalar('w_h_loss', w_h_loss.item(), step)
    return loss.item()


print('# generator parameters:', sum(param.numel() for param in model.parameters()))
print('Starting training...')
loss = 100  # max loss
for epoch in range(1, epoches + 1):
    c_loss = train(epoch)
    lr_scheduler.step(epoch)

    if epoch % 5 == 0 and c_loss < loss:
        print("at epoch: {} saving model for loss: {} to loss: {}....".format(epoch, loss, c_loss))
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoches}
        torch.save(state, 'chkpt-smallhourglass-0627-1.pt')
        loss = c_loss