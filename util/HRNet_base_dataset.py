# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
import random

import torch
from torch.nn import functional as F
from torch.utils import data

# from config import config


class inference_engine(data.Dataset):
    def __init__(self,
                 ignore_label=255,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1./downsample_rate

        self.files = []

    def __len__(self):
        return len(self.files)

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        return np.array(label).astype('int32')

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    def multi_scale_aug(self, image, label=None,
                        rand_scale=1, rand_crop=True):
        # long_size = np.int(self.base_size * rand_scale + 0.5)
        long_size = int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            # new_w = np.int(w * long_size / h + 0.5)
            new_w = int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            # new_h = np.int(h * long_size / w + 0.5)
            new_h = int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label = self.rand_crop(image, label)

        return image, label

    def resize_short_length(self, image, label=None, short_length=None, fit_stride=None, return_padding=False):
        h, w = image.shape[:2]
        if h < w:
            new_h = short_length
            new_w = np.int(w * short_length / h + 0.5)
        else:
            new_w = short_length
            new_h = np.int(h * short_length / w + 0.5)        
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = 0, 0
        if fit_stride is not None:
            pad_w = 0 if (new_w % fit_stride == 0) else fit_stride - (new_w % fit_stride)
            pad_h = 0 if (new_h % fit_stride == 0) else fit_stride - (new_h % fit_stride)
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, 
                cv2.BORDER_CONSTANT, value=tuple(x * 255 for x in self.mean[::-1])
            )

        if label is not None:
            label = cv2.resize(
                label, (new_w, new_h),
                interpolation=cv2.INTER_NEAREST)
            if pad_h > 0 or pad_w > 0:
                label = cv2.copyMakeBorder(
                    label, 0, pad_h, 0, pad_w, 
                    cv2.BORDER_CONSTANT, value=self.ignore_label
                )
            if return_padding:
                return image, label, (pad_h, pad_w)
            else:
                return image, label
        else:
            if return_padding:
                return image, (pad_h, pad_w)
            else:
                return image  

    def random_brightness(self, config, img):
        if not config.TRAIN.RANDOM_BRIGHTNESS:
            return img
        if random.random() < 0.5:
            return img
        self.shift_value = config.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE
        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def gen_sample(self, image, label,
                   multi_scale=True, is_flip=True):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label,
                                                rand_scale=rand_scale)

        image = self.random_brightness(image)
        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(
                label,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )

        return image, label

    def reduce_zero_label(self, labelmap):
        labelmap = np.array(labelmap)
        encoded_labelmap = labelmap - 1

        return encoded_labelmap

    def inference(self, model, image, flip=False):
        size = image.size()  # 1, 3, H, W
        pred = model(image)

        # if config.MODEL.NUM_OUTPUTS > 1:
        #     pred = pred[config.TEST.OUTPUT_INDEX]

        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=True
        )

        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))

            # if config.MODEL.NUM_OUTPUTS > 1:
            #     flip_output = flip_output[config.TEST.OUTPUT_INDEX]

            flip_output = F.interpolate(
                input=flip_output, size=size[-2:],
                mode='bilinear', align_corners=True
            )

            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(
                flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()
    
    # 这个利用 E:\Code_Python\PYTHON_code\Semantic_segmentation\7_HRNet+OCR\lib\datasets\cityscapes.py 中的 multi_scale_inference
    def multi_scale_inference(self, cfg, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."

        # 将 3xHxW ----> HxWx3 是为了使用 cv2.resize
        image = image.numpy()[0].transpose((1,2,0)).copy()   # 3xHxW ----> HxWx3。

        # 这里 PSP 中用的是 stride_rate=2/3 的步长尺寸
        stride_rate=2/3
        stride_h = np.int(self.crop_size[0] * stride_rate)   # 这里应该是 slid 的步长
        stride_w = np.int(self.crop_size[1] * stride_rate)
        final_pred = torch.zeros([1, cfg['num_classes'],
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, cfg['num_classes'],
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width), 
                mode='bilinear', align_corners=True
            )            
            final_pred += preds
        return final_pred
    
    # 根据 PSP 代码改良版的 
    def multi_scale_inference_PSP(self, cfg, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()   # 1, 3, H, W
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy() # 3xHxW ----> HxWx3。

        # 这里 PSP 中用的是 stride_rate=2/3 的步长尺寸
        # stride_rate=2/3
        stride_rate = cfg['stride_rate']

        # stride_h = np.int(self.crop_size[0] * stride_rate)   # 这里应该是 slid 的步长
        # stride_w = np.int(self.crop_size[1] * stride_rate)
        stride_h = int(self.crop_size[0] * stride_rate)   # 这里应该是 slid 的步长
        stride_w = int(self.crop_size[1] * stride_rate)

        final_pred = torch.zeros([1, cfg['num_classes'],
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            # ---------------------------------------------------------------------------
            # 如果 scale 后的图像小于 crop_h, crop_w，进行 0 填充
            pad_h = max(self.crop_size[0] - height, 0)   # 需要进行 0 填充的个数
            pad_w = max(self.crop_size[1] - width, 0)
            pad_h_half = int(pad_h / 2)
            pad_w_half = int(pad_w / 2)
            if pad_h > 0 or pad_w > 0:
                # image 的四周进行了 0 padding
                new_img = cv2.copyMakeBorder(new_img, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=cfg['mean'])

            # ---------------------------------------------------------
            # 这里注意，不分 scales 只要 crop_size 小于 scales 后的 image 都要进行分块的 inference
            new_h, new_w = new_img.shape[:-1]  # scale 后，并且可能进行了 0 padding 的新图像的尺寸
            
            # -------------------------------------
            # 计算滑窗个数
            # 向上取整，并加 1
            rows = int(np.ceil(1.0 * (new_h - 
                            self.crop_size[0]) / stride_h)) + 1
            cols = int(np.ceil(1.0 * (new_w - 
                            self.crop_size[1]) / stride_w)) + 1
            preds = torch.zeros([1, cfg['num_classes'],
                                        new_h,new_w]).cuda()
            count = torch.zeros([1,1, new_h, new_w]).cuda()

            for r in range(rows):
                for c in range(cols):
                    h0 = r * stride_h
                    w0 = c * stride_w
                    h1 = min(h0 + self.crop_size[0], new_h)
                    w1 = min(w0 + self.crop_size[1], new_w)
                    h0 = max(int(h1 - self.crop_size[0]), 0)
                    w0 = max(int(w1 - self.crop_size[1]), 0)
                    crop_img = new_img[h0:h1, w0:w1, :]
                    crop_img = crop_img.transpose((2, 0, 1))      # HxWx3 --> 3xHxW
                    crop_img = np.expand_dims(crop_img, axis=0)
                    crop_img = torch.from_numpy(crop_img)
                    pred = self.inference(model, crop_img, flip)
                    preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                    count[:,:,h0:h1,w0:w1] += 1
            preds = preds / count

            preds = preds[:,:,pad_h_half:pad_h_half+height,pad_w_half:pad_w_half+width]   # 去除掉 padding 的边界

            preds = F.interpolate(
                preds, (ori_height, ori_width), 
                mode='bilinear', align_corners=True
            )            
            final_pred += preds
        return final_pred