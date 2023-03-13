import torch
import numpy as np
from PIL import Image
from typing import Optional



class dynamic_resize:
    # base_size: h x w
    def __init__(self, L=1280, base_size=[384, 384]):
        self.L = L
        self.base_size = base_size[::-1]

    def __call__(self, sample):
        size = list(sample['image'].size)
        if (size[0] >= size[1]) and size[1] > self.L:
            size[0] = size[0] / (size[1] / self.L)
            size[1] = self.L
        elif (size[1] > size[0]) and size[0] > self.L:
            size[1] = size[1] / (size[0] / self.L)
            size[0] = self.L
        size = (int(round(size[0] / 32)) * 32, int(round(size[1] / 32)) * 32)

        if 'image' in sample.keys():
            sample['image_resized'] = sample['image'].resize(self.base_size, Image.BILINEAR)
            sample['image'] = sample['image'].resize(size, Image.BILINEAR)

        if 'gt' in sample.keys():
            sample['gt_resized'] = sample['gt'].resize(self.base_size, Image.NEAREST)
            sample['gt'] = sample['gt'].resize(size, Image.NEAREST)

        return sample


class tonumpy:
    def __init__(self):
        pass

    def __call__(self, sample):
        for key in sample.keys():
            if key in ['image', 'image_resized', 'gt', 'gt_resized']:
                sample[key] = np.array(sample[key], dtype=np.float32)

        return sample


class normalize:
    def __init__(self, mean: Optional[list] = None, std: Optional[list] = None, div=255):
        self.mean = mean if mean is not None else 0.0
        self.std = std if std is not None else 1.0
        self.div = div

    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] /= self.div
            sample['image'] -= self.mean
            sample['image'] /= self.std

        if 'image_resized' in sample.keys():
            sample['image_resized'] /= self.div
            sample['image_resized'] -= self.mean
            sample['image_resized'] /= self.std

        if 'gt' in sample.keys():
            sample['gt'] /= self.div

        if 'gt_resized' in sample.keys():
            sample['gt_resized'] /= self.div

        return sample


class totensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] = sample['image'].transpose((2, 0, 1))
            sample['image'] = torch.from_numpy(sample['image']).float()

        if 'image_resized' in sample.keys():
            sample['image_resized'] = sample['image_resized'].transpose((2, 0, 1))
            sample['image_resized'] = torch.from_numpy(sample['image_resized']).float()

        if 'gt' in sample.keys():
            sample['gt'] = torch.from_numpy(sample['gt'])
            sample['gt'] = sample['gt'].unsqueeze(dim=0)

        if 'gt_resized' in sample.keys():
            sample['gt_resized'] = torch.from_numpy(sample['gt_resized'])
            sample['gt_resized'] = sample['gt_resized'].unsqueeze(dim=0)

        return sample
