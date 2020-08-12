import torch
from torch.utils.data import DataLoader
import time
import numpy as np
import pandas as pd

# self_written .py file
import utils
import visualizeHelper
import models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_csv = 'global-wheat-detection/train.csv'
test_csv = 'global-wheat-detection/submission.csv'

train_dir = 'global-wheat-detection/images/train'
test_dir = 'global-wheat-detection/images/test'

def train(pretrained = True):
    train_df, val_df = utils.process_csv(train_csv)

    train_set = utils.Wheatset(train_df, train_dir, phase='train')
    val_set = utils.Wheatset(val_df, train_dir, phase='validation')

    # batching
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_data_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
        val_set,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )


    # images, targets, ids = next(iter(train_data_loader))
    # images = list(image.to(device) for image in images)
    # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # construct fasterrcnn network
    model = models.construct_models()
    if pretrained:
        WEIGHTS_FILE = '/checkpoints/bestmodel_may28.pt'
        weights = torch.load(WEIGHTS_FILE)
        model.load_state_dict(weights['state_dict'])

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    #train
    num_epochs = 5
    train_loss_min = 0.9
    total_train_loss = []

    checkpoint_path = '/checkpoints/chkpoint_'
    best_model_path = '/checkpoints/bestmodel_may28.pt'

    for epoch in range(num_epochs):
        print(f'Epoch :{epoch + 1}')
        start_time = time.time()
        train_loss = []
        model.train()
        for images, targets, image_ids in train_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            train_loss.append(losses.item())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        # train_loss/len(train_data_loader.dataset)
        epoch_train_loss = np.mean(train_loss)
        total_train_loss.append(epoch_train_loss)
        print(f'Epoch train loss is {epoch_train_loss}')

        #     if lr_scheduler is not None:
        #         lr_scheduler.step()

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'train_loss_min': epoch_train_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        utils.save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        ## TODO: save the model if validation loss has decreased
        if epoch_train_loss <= train_loss_min:
            print('Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min, epoch_train_loss))
            # save checkpoint as best model
            utils.save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            train_loss_min = epoch_train_loss

        time_elapsed = time.time() - start_time
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def test():
    model = models.construct_models()

    WEIGHTS_FILE = '/checkpoints/bestmodel_may28.pt'
    weights = torch.load(WEIGHTS_FILE)
    model.load_state_dict(weights['state_dict'])

    model.to(device)


    test_df = pd.read_csv(test_csv)
    transform = utils.get_transforms('test')
    test_set = utils.WheatTestDataset(test_df, test_dir, transform)

    def collate_fn(batch):
        return tuple(zip(*batch))

    test_loader = DataLoader(
        test_set,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn
    )

    detection_threshold = 0.5
    model.eval()

    for images, image_ids in test_loader:

        images = list(image.to(device) for image in images)
        outputs = model(images)

        for i, image in enumerate(images):
            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()

            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            image_id = image_ids[i]

            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            visualizeHelper.vis_boxes(image, boxes, scores)