import torch
import os
import argparse
import numpy as np
import torch.utils.data as data
import scipy.misc
import imageio
from torchvision import transforms
import skimage.morphology as sm
from model import Cascade_AttenUnet
from utils.loadData import ISBITestDataset
from tqdm import tqdm

from utils.save_tiff import image_array_to_tiff, transfer_tif


parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
parser.add_argument('--save_path', type=str, default='./results')
parser.add_argument('--experiment_idx', type=int, default=0)
parser.add_argument('--epoch_idx', type=int, default=1550)
parser.add_argument('--gamma', type=int, default=3)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = os.path.join(args.checkpoint_dir, str(args.experiment_idx))
out_dir = os.path.join(args.save_path, str(args.experiment_idx))

if os.path.exists(checkpoint_dir) is not True:
    print("something wrong")
    print("&&&"*20)

def save_img(img, mask, idx, save_path):

    # mask = mask > threshold

    #
    # mask = sm.closing(mask)
    img = img[64:576, 64:576]
    mask = mask[64:576, 64:576]
    img = (img + 1) / 2
    mask = (mask + 1) / 2
    imageio.imsave(os.path.join(save_path, "%d-img.png" % idx), img)
    imageio.imsave(os.path.join(save_path, "%d-label.png" % idx), mask)

    # stage1 = stage1[64:576, 64:576]
    # stage2 = stage2[64:576, 64:576]
    # stage3 = stage3[64:576, 64:576]
    #
    # stage1 = (stage1 + 1) / 2
    # stage2 = (stage2 + 1) / 2
    # stage3 = (stage3 + 1) / 2
    # scipy.misc.imsave(os.path.join(save_path, "%d-stage1.png" % idx), stage1)
    # scipy.misc.imsave(os.path.join(save_path, "%d-stage2.png" % idx), stage2)
    # scipy.misc.imsave(os.path.join(save_path, "%d-stage3.png" % idx), stage3)

def save_img_1(img, mask, idx, save_path):
    img = img[64:576, 64:576]
    mask = mask[64:576, 64:576]
    img = (img + 1) / 2
    mask = (mask + 1) / 2
    imageio.imsave(os.path.join(save_path, "%d-img.png" % idx), img)
    imageio.imsave(os.path.join(save_path, "%d-label.png" % idx), mask)



def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])



    dataset = ISBITestDataset('/home/guoyuze/lmser_seg_2228/Dataset/ISBI2012/test_img_1', transform=transform)

    model = Cascade_AttenUnet(init_depth=12, gamma=args.gamma)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'epoch-%d.pth' % args.epoch_idx)))
    model.to(device)

    dataloader = data.DataLoader(dataset, batch_size=1)

    test_results_dir = os.path.join(out_dir, "testing_results")
    if os.path.exists(test_results_dir) is not True:
        os.makedirs(test_results_dir)

    test_results_dir_stage1 = os.path.join(out_dir, "testing_results_stage1")
    if os.path.exists(test_results_dir_stage1) is not True:
        os.makedirs(test_results_dir_stage1)

    test_results_dir_stage2 = os.path.join(out_dir, "testing_results_stage2")
    if os.path.exists(test_results_dir_stage2) is not True:
        os.makedirs(test_results_dir_stage2)

    for idx, (test_img, img_number) in tqdm(enumerate(dataloader), total=len(dataloader)):
        test_img = test_img.float().to(device)
        pred_label, pred_stage1, pred_stage2 = model(test_img)

        img = torch.squeeze(test_img)
        img = img.to("cpu").detach().numpy()

        mask = torch.squeeze(pred_label).to("cpu").detach().numpy()

        mask_stage1 = torch.squeeze(pred_stage1).to("cpu").detach().numpy()
        mask_stage2 = torch.squeeze(pred_stage2).to("cpu").detach().numpy()

        """
        img.shape (512, 512)
        mask.shape (512, 512)
        """
        save_img(img, mask, idx, test_results_dir)
        save_img(img, mask_stage1, idx, test_results_dir_stage1)
        save_img(img, mask_stage2, idx, test_results_dir_stage2)
    image_array_to_tiff('probabilities_test1.tif', test_results_dir)
    transfer_tif(os.path.join(test_results_dir, 'probabilities_test1.tif'), os.path.join(test_results_dir, 'probabilities_test.tif'))

    image_array_to_tiff('probabilities_test1_stage1.tif', test_results_dir_stage1)
    transfer_tif(os.path.join(test_results_dir_stage1, 'probabilities_test1_stage1.tif'),
                 os.path.join(test_results_dir_stage1, 'probabilities_test_stage1.tif'))

    image_array_to_tiff('probabilities_test1_stage2.tif', test_results_dir_stage2)
    transfer_tif(os.path.join(test_results_dir_stage2, 'probabilities_test1_stage2.tif'),
                 os.path.join(test_results_dir_stage2, 'probabilities_test_stage2.tif'))

    gen_train_dir = os.path.join(out_dir, 'training_results')
    if os.path.exists(gen_train_dir) is not True:
        os.makedirs(gen_train_dir)
    gen_train_dataset = ISBITestDataset('/home/guoyuze/lmser_seg_2228/Dataset/ISBI2012/train_img_full', transform=transform)
    gen_train_dataloader = data.DataLoader(gen_train_dataset, batch_size=1)
    for idx, (test_img, img_number) in tqdm(enumerate(gen_train_dataloader), total=len(dataloader)):
        test_img = test_img.float().to(device)
        pred_label, _, _ = model(test_img)
        img = torch.squeeze(test_img)
        img = img.to("cpu").detach().numpy()
        mask = torch.squeeze(pred_label).to("cpu").detach().numpy()

        """
        img.shape (512, 512)
        mask.shape (512, 512)
        """
        save_img(img, mask, idx, gen_train_dir)
    image_array_to_tiff('probabilities_train1.tif', gen_train_dir)
    transfer_tif(os.path.join(gen_train_dir, 'probabilities_train1.tif'), os.path.join(gen_train_dir, 'probabilities_train.tif'))

if __name__ == '__main__':
    main()



