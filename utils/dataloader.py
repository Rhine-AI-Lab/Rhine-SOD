import os
import cv2
import sys

import numpy as np
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from PIL import Image
from threading import Thread

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.custom_transforms import *
from utils.misc import *

Image.MAX_IMAGE_PIXELS = None


def get_transform(tfs):
    comp = []
    for key, value in zip(tfs.keys(), tfs.values()):
        if value is not None:
            tf = eval(key)(**value)
        else:
            tf = eval(key)()
        comp.append(tf)
    return transforms.Compose(comp)


class RGB_Dataset(Dataset):
    def __init__(self, root, sets, tfs):
        self.images, self.gts = [], []

        for set in sets:
            image_root, gt_root = os.path.join(root, set, 'images'), os.path.join(root, set, 'masks')

            images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                      f.lower().endswith(('.jpg', '.png'))]
            images = sort(images)

            gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png'))]
            gts = sort(gts)

            self.images.extend(images)
            self.gts.extend(gts)

        self.filter_files()

        self.size = len(self.images)
        self.transform = get_transform(tfs)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        shape = gt.size[::-1]
        name = self.images[index].split(os.sep)[-1]
        name = os.path.splitext(name)[0]

        sample = {'image': image, 'gt': gt, 'name': name, 'shape': shape}

        sample = self.transform(sample)
        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img, gt = Image.open(img_path), Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images, self.gts = images, gts

    def __len__(self):
        return self.size


class ImageLoader:
    def __init__(self, root, tfs):
        if os.path.isdir(root):
            self.images = [os.path.join(root, f) for f in os.listdir(root) if
                           f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            self.images = sort(self.images)
        elif os.path.isfile(root):
            self.images = [root]
        self.size = len(self.images)
        self.transform = get_transform(tfs)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == self.size:
            raise StopIteration
        image = Image.open(self.images[self.index]).convert('RGB')
        shape = image.size[::-1]
        name = self.images[self.index].split(os.sep)[-1]
        name = os.path.splitext(name)[0]

        sample = {'image': image, 'name': name, 'shape': shape, 'original': image}
        sample = self.transform(sample)
        sample['image'] = sample['image'].unsqueeze(0)
        if 'image_resized' in sample.keys():
            sample['image_resized'] = sample['image_resized'].unsqueeze(0)

        self.index += 1
        return sample

    def __len__(self):
        return self.size


class RefinementLoader:
    def __init__(self, image_dir, seg_dir, tfs):
        self.images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                       f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.images = sort(self.images)

        self.segs = [os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if
                     f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.segs = sort(self.segs)

        self.size = len(self.images)
        self.transform = get_transform(tfs)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == self.size:
            raise StopIteration
        image = Image.open(self.images[self.index]).convert('RGB')
        seg = Image.open(self.segs[self.index]).convert('L')
        shape = image.size[::-1]
        name = self.images[self.index].split(os.sep)[-1]
        name = os.path.splitext(name)[0]

        sample = {'image': image, 'gt': seg, 'name': name, 'shape': shape, 'original': image}
        sample = self.transform(sample)
        sample['image'] = sample['image'].unsqueeze(0)
        sample['mask'] = sample['gt'].unsqueeze(0)
        if 'image_resized' in sample.keys():
            sample['image_resized'] = sample['image_resized'].unsqueeze(0)
        del sample['gt']

        self.index += 1
        return sample

    def __len__(self):
        return self.size
