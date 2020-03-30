import torch
import math
import random
import numpy as np
import numbers
from PIL import Image, ImageOps


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.padding = padding

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            label = ImageOps.expand(label, border=self.padding, fill=0)

        # w, h = img.shape[0], img.shape[1]
        w, h = img.size
        tw, th = self.size
        if w==tw and h==th:
            return {'image': img, 'label': label}

        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            label = label.resize((tw, th), Image.NEAREST)
            return {'image': img, 'label': label}

        x1 = random.randint(0, w-tw)
        y1 = random.randint(0, h-th)
        img = img.crop((x1, y1, x1+tw, y1+th))
        label = label.crop((x1, y1, x1+tw, y1+th))
        return {'image': img, 'label': label}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'label': label}


class ToTensor(object):
    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        label = np.expand_dims(np.array(sample['label']), -1).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()
        label = torch.squeeze(label, 0)

        return {'image': img, 'label': label}

class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        label = np.array(sample['label'])

        img /= 255.0
        img -= self.mean
        img /=self.std

        return {'image': img, 'label': label}



