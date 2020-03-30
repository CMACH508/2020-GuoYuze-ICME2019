import torch
import scipy.misc
import imageio
import torch.utils.data as data
from torchvision import transforms
import os
import numpy as np
from skimage.transform import rotate
from skimage.transform import resize
from skimage import exposure
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def elastic_transform(image, label, alpha, sigma, random_state=None):
    if random_state == None:
        random_state = np.random.RandomState(None)


    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    dz = np.zeros_like(dx)

    # x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    # print(x.shape)
    # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    distored_label = map_coordinates(label, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape), distored_label.reshape(label.shape)


class ISBITrainDataset(data.Dataset):

    def __init__(self, img_dir, transform=None):
        self.train_img_dir = os.path.join(img_dir, 'train_img')
        self.train_label_dir = os.path.join(img_dir, 'train_label')
        self.transform = transform

    def __getitem__(self, idx):
        img = imageio.imread(os.path.join(self.train_img_dir, '%d.png' % idx))
        label = imageio.imread(os.path.join(self.train_label_dir, '%d.png' % idx))



        ## random crop
        crop_size = 448
        x = random.randint(0, 512 - crop_size)
        y = random.randint(0, 512 - crop_size)
        img = img[x:x + crop_size, y:y + crop_size]
        label = label[x:x + crop_size, y:y + crop_size]

        ## flip an image
        seed1 = random.random()
        if seed1 > 0.5:
            img = img[:,::-1]
            label = label[:,::-1]

        ## rotate
        angle = random.random()
        img = rotate(img, 90*int(angle/0.25))
        label = rotate(label, 90*int(angle/0.25))

        ## elastic_transform
        seed = random.random()
        if seed > 0.5:
            # img = elastic_transform(img, alpha=300, sigma=15)
            # label = elastic_transform(label, alpha=300, sigma=15)
            img, label = elastic_transform(img, label, alpha=300, sigma=15)

        ## Overlap-tile
        lap_size = 64
        img = np.pad(img, ((lap_size, lap_size), (lap_size, lap_size)), 'reflect')
        label = np.pad(label, ((lap_size, lap_size), (lap_size, lap_size)), 'reflect')

        # [0, 1] => [-1, 1]
        img = img * 2 - 1
        label = label * 2 - 1

        img = np.expand_dims(img, 2)
        label = np.expand_dims(label, 2)
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(os.listdir(self.train_img_dir))

class ISBIEvalDataset(data.Dataset):

    def __init__(self, img_dir, transform=None):
        self.train_img_dir = os.path.join(img_dir, 'test_img')
        self.train_label_dir = os.path.join(img_dir, 'test_label')
        # self.img_path = list(map(lambda x: os.path.join(self.train_img_dir, x), os.listdir(self.train_img_dir)))
        # self.label_path = list(map(lambda x: os.path.join(self.train_label_dir, x), os.listdir(self.train_label_dir)))
        self.transform = transform

    def __getitem__(self, idx):
        img = imageio.imread(os.path.join(self.train_img_dir, '%d.png' % idx))
        label = imageio.imread(os.path.join(self.train_label_dir, '%d.png' % idx))

        ## flip an image
        seed1 = random.random()
        if seed1 > 0.5:
            img = img[:,::-1]
            label = label[:,::-1]

        ## rotate
        angle = random.random()
        img = rotate(img, 90*int(angle/0.25))
        label = rotate(label, 90*int(angle/0.25))

        # [0, 1] => [-1, 1]
        img = img * 2 - 1
        label = label * 2 - 1

        img = np.expand_dims(img, 2)
        label = np.expand_dims(label, 2)
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)



        return img, label

    def __len__(self):
        return len(os.listdir(self.train_img_dir))

class ISBITestDataset(data.Dataset):

    def __init__(self, img_dir, transform=None):
        # self.test_img_dir = os.path.join(img_dir, 'test_img_1')
        self.test_img_dir = img_dir
        self.transform = transform

    def __getitem__(self, idx):
        img = imageio.imread(os.path.join(self.test_img_dir, '%d.png' % idx))

        lap_size = 64
        img = np.pad(img, ((lap_size, lap_size), (lap_size, lap_size)), 'reflect')

        # [0, 1] => [-1, 1]
        img = img / 255
        img = img * 2.0 - 1.0

        img = np.expand_dims(img, 2)
        if self.transform is not None:
            img = self.transform(img)

        return img, idx

    def __len__(self):
        return len(os.listdir(self.test_img_dir))

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5), (0.5, 0.5))
    ])
    # dataset = ISBITestDataset('/home/guoyuze/lmser_seg/Dataset/ISBI2012/', transform=transform)
    # dataset = NucleiDataset('/home/guoyuze/lmser_seg/Dataset/nuclei/train/', transform=transform)
    # dataset = NucleiTestDataset('/home/guoyuze/lmser_seg/Dataset/nuclei/test/', transform=transform)
    dataset = ISBITrainDataset_cv('/home/guoyuze/lmser_seg/Dataset/ISBI2012/', cv_idx=5, transform=transform)
    # dataset = ISBITestDataset('/home/guoyuze/lmser_seg/Dataset/ISBI2012/', transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)

    iters = iter(dataloader)
    count = 0
    while True:
        imgs, masks = next(iters)
        print(imgs.size())
        print(torch.max(imgs))
        print(torch.min(imgs))
        print(masks.size())
        print(torch.max(masks))
        print(torch.min(masks))

    # imgs, masks = next(iters)
    # print(imgs.size())
    # print(torch.max(imgs))
    # print(torch.min(imgs))
    # print(masks.size())
    # print(torch.max(masks))
    # print(torch.min(masks))



