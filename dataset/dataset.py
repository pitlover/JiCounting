from typing import Dict, Tuple, Optional, Any
import cv2
import json
import numpy as np
import torch
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class FSC_Dataset(Dataset):
    def __init__(self,
                 data_path: str,
                 data_type: str = "FSC147",
                 mode: str = "train"
                 ):
        super().__init__()
        MIN_HW = 384
        MAX_HW = 1584
        self.mode = mode.lower()
        self.data_type = data_type.upper()

        if self.mode not in ("train", "val", "test"):
            raise ValueError(f"JiCounting mode {mode} is not supported.")

        if self.data_type not in ("FSC147"):
            raise ValueError(f"JiCounting data_type {data_type} is not supported.")

        self.anno_file = join(data_path, 'annotation_FSC147_384.json')
        self.data_split_file = join(data_path, 'Train_Test_Val_FSC_147.json')
        self.im_dir = join(data_path, 'images_384_VarV2')
        self.gt_dir = join(data_path, 'gt_density_map_adaptive_384_VarV2')

        with open(self.anno_file) as f:
            self.annotations = json.load(f)

        with open(self.data_split_file) as f:
            self.data_split = json.load(f)

        self.id_list = self.data_split[mode]
        self.Transform_train = transforms.Compose([resizeImageWithGT(MAX_HW)])
        self.Transform_test = transforms.Compose([resizeImage(MAX_HW)])

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, index: int):
        id = self.id_list[index]
        anno = self.annotations[id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(self.im_dir, id))
        density_path = self.gt_dir + '/' + id.split(".jpg")[0] + ".npy"
        density = np.load(density_path).astype('float32')

        if self.mode == 'train':
            sample = {'image': image, 'lines_boxes': rects[:3], 'gt_density': density}
            sample = self.Transform_train(sample)
            return sample['image'], sample['boxes'], sample['gt_density']
        else:
            sample = {'image': image, 'lines_boxes': rects[:3], 'gt_density': density, 'dots': dots}
            sample = self.Transform_test(sample)
            sample['gt_cnt'] = dots.shape[0]
            return sample['image'], sample['boxes'], sample['gt_cnt'], sample['gt_density']


class resizeImageWithGT(object):
    def __init__(self, MAX_HW: int = 1504, MIN_HW: int = 384):
        self.max_hw = MAX_HW
        self.min_hw = MIN_HW
        self.hw = 512

    def __call__(self, sample):
        image, lines_boxes, density = sample['image'], sample['lines_boxes'], sample['gt_density']
        W, H = image.size  ## y, x

        Normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if W != self.hw or H != self.hw:
            scale_factor_H, scale_factor_W = float(self.hw) / H, float(self.hw) / W
            new_H, new_W = self.hw, self.hw
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)
            if new_count > 0: resized_density = resized_density * (orig_count / new_count)
        else:
            resized_image = image
            resized_density = density

        boxes = list()

        for box in lines_boxes:
            y1, x1, y2, x2 = int(box[0] * scale_factor_H), int(box[1] * scale_factor_W), int(
                box[2] * scale_factor_H), int(box[3] * scale_factor_W)
            boxes.append([0, y1, x1, y2, x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0)

        sample = {'image': resized_image, 'boxes': boxes, 'gt_density': resized_density}
        return sample


class resizeImage(object):
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW
        self.hw = 512

    def __call__(self, sample):
        image, lines_boxes, density, lines_dots = sample['image'], sample['lines_boxes'], sample['gt_density'], sample[
            'dots']

        W, H = image.size

        Normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if W != self.hw or H != self.hw:
            scale_factor_H, scale_factor_W = float(self.hw) / H, float(self.hw) / W
            new_H, new_W = self.hw, self.hw
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)
            if new_count > 0: resized_density = resized_density * (orig_count / new_count)
        else:
            resized_image = image
            resized_density = density

        boxes = list()
        dots = list()
        for box in lines_boxes:
            y1, x1, y2, x2 = int(box[0] * scale_factor_H), int(box[1] * scale_factor_W), int(
                box[2] * scale_factor_H), int(box[3] * scale_factor_W)
            boxes.append([0, y1, x1, y2, x2])

        for dot in lines_dots:
            y1, x1 = int(dot[0] * scale_factor_W), int(dot[1] * scale_factor_H)
            dots.append([y1, x1])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0)

        sample = {'image': resized_image, 'boxes': boxes, 'dots': dots, 'gt_density': resized_density}
        return sample
