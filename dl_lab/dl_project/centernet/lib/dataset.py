import torch
import numpy as np
import os
import cv2
import lib.utils as utils



COCO_MEAN = [0.40789654, 0.44719302, 0.47026115]
COCO_STD = [0.28863828, 0.27408164, 0.27809835]
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]


class WheatTest(torch.utils.data.Dataset):
    '''
    return [img, hmap, _w_h, regs, indx, ind_mask, center, scale, img_id]
    '''

    def __init__(self, dataframe, data_dir, fix_size=512):
        super(WheatTest, self).__init__()
        self.num_classes = 1
        self.data_dir = data_dir
        self.fix_size = fix_size

        self.data_rng = np.random.RandomState(123)
        self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]

        self.df = dataframe
        self.ids = dataframe['image_id'].unique()

        self.max_objs = 128
        self.padding = 31  # 31 for resnet/resdcn
        self.down_ratio = 4
        self.img_size = {'h': fix_size, 'w': fix_size}
        self.fmap_size = {'h': fix_size // self.down_ratio, 'w': fix_size // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.7

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.data_dir, 'test', self.ids[idx] + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img = cv2.resize(img,(self.fix_size,self.fix_size)) # convert to fix_size, default by 512
        height, width = img.shape[0], img.shape[1]
        center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image

        scale = max(height, width) * 1.0

        trans_img = utils.get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']])

        img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h']))

        img = img.astype(np.float32) / 255.

        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

        return {'image': img, 'c': center, 's': scale, 'img_id': img_id}

    def __len__(self):
        return len(self.ids)


class Wheat(torch.utils.data.Dataset):
    '''
    return [img, hmap, _w_h, regs, indx, ind_mask, center, scale, img_id]
    '''


    def __init__(self, dataframe, data_dir, train=True, transform=None, fix_size=512):
        super(Wheat, self).__init__()
        self.num_classes = 1
        self.transform = transform
        self.data_dir = data_dir
        self.fix_size = fix_size

        self.data_rng = np.random.RandomState(123)
        self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
        self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]

        self.df = dataframe
        self.ids = dataframe['image_id'].unique()
        self.train = train

        self.max_objs = 128
        self.padding = 31  # 31 for resnet/resdcn
        self.down_ratio = 4
        self.img_size = {'h': fix_size, 'w': fix_size}
        self.fmap_size = {'h': fix_size // self.down_ratio, 'w': fix_size // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.7

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.data_dir, self.ids[idx] + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img = cv2.resize(img,(self.fix_size,self.fix_size)) # convert to fix_size, default by 512
        height, width = img.shape[0], img.shape[1]
        center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image

        scale = max(height, width) * 1.0

        flipped = False
        if self.train:
            scale = scale * np.random.choice(self.rand_scales)
            w_border = self._get_border(128, img.shape[1])
            h_border = self._get_border(128, img.shape[0])
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)

            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                center[0] = width - center[0] - 1

        trans_img = utils.get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']])

        img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h']))

        annos = self.df[self.df['image_id'].isin([self.ids[idx]])]

        bboxes = annos[['x', 'y', 'w', 'h']].values
        bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy
        gt = bboxes.copy()
        labels = np.zeros(len(bboxes)).astype(np.uint8)
        img = img.astype(np.float32) / 255.

        if self.train:
            utils.color_aug(self.data_rng, img, self.eig_val, self.eig_vec)


        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

        trans_fmap = utils.get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']])
        hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

        #         ###
        #         fig = plt.figure()

        #         ax1 = fig.add_subplot(121)
        #         ax1.imshow(trans_img)

        #         ax2 = fig.add_subplot(122)
        #         ax2.imshow(trans_fmap)
        #         ###
        for k, (bbox, label) in enumerate(zip(bboxes, labels)):
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = utils.affine_transform(bbox[:2], trans_fmap)
            bbox[2:] = utils.affine_transform(bbox[2:], trans_fmap)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if h > 0 and w > 0:
                obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                obj_c_int = obj_c.astype(np.int32)

                radius = max(0, int(utils.gaussian_radius((np.ceil(h), np.ceil(w)), self.gaussian_iou)))
                utils.draw_umich_gaussian(hmap[label], obj_c_int, radius)
                w_h_[k] = 1. * w, 1. * h
                regs[k] = obj_c - obj_c_int  # discretization error
                inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
                ind_masks[k] = 1
        #         ###
        #         fig = plt.figure()

        #         ax1 = fig.add_subplot(121)
        #         ax1.imshow(img.transpose(1,2,0))

        #         ax2 = fig.add_subplot(122)
        #         ax2.imshow(hmap.transpose(1,2,0).squeeze(2))
        #         ###

        return {'image': img,
                'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
                'c': center, 's': scale, 'img_id': img_id,'boxes':gt}

    def __len__(self):
        return len(self.ids)