import numpy as np
import scipy.misc
import os
import shutil

"""
data process for dataset Fluo-N2DH-GOWT1
"""
# PATH = '../Dataset/CellTracking/training-datasets/Fluo-N2DH-GOWT1/02_GT/SEG'
# img_name = os.listdir(PATH)
#
# for img in img_name:
#     save_name = img.split('.')[0] + '.jpg'
#     img_path = os.path.join(PATH, img)
#     a = scipy.misc.imread(img_path)
#     a = a / np.max(a)
#     scipy.misc.imsave(os.path.join(PATH, save_name), a)


"""
dataset nuclei
"""
def process_nuclei():
    PATH = '../Dataset/nuclei/stage1_train'
    SAVE_PATH = '../Dataset/nuclei/data'
    SAVE_PATH_IMG = os.path.join(SAVE_PATH, 'imgs')
    SAVE_PATH_MASK = os.path.join(SAVE_PATH, 'masks')
    if os.path.exists(SAVE_PATH) is False:
        os.makedirs(SAVE_PATH)
        os.makedirs(SAVE_PATH_IMG)
        os.makedirs(SAVE_PATH_MASK)

    pairs = os.listdir(PATH)
    for idx, pair in enumerate(pairs):
        pair_path = os.path.join(PATH, pair)
        img_path = os.path.join(pair_path, 'images')
        img_name = os.listdir(img_path)
        print(idx)
        print(img_name)
        shutil.copy(os.path.join(img_path, img_name[0]), os.path.join(SAVE_PATH_IMG, "%d.png" % idx))

        mask_path = os.path.join(pair_path, 'masks')
        mask_names = os.listdir(mask_path)
        shape = scipy.misc.imread(os.path.join(img_path, img_name[0])).shape
        print(shape)
        mask = np.zeros(shape[:2])
        for mask_name in mask_names:
            mask = mask + scipy.misc.imread(os.path.join(mask_path, mask_name))

        scipy.misc.imsave(os.path.join(SAVE_PATH_MASK, '%d.png' % idx), mask)

def move_files():
    path = '../Dataset/nuclei/test'
    save_path = '../Dataset/nuclei/train'
    sample_names = os.listdir(path)
    for sample_name in sample_names:
        sample_path = os.path.join(path, sample_name)
        img_path = os.path.join(sample_path, 'images')
        img_name = os.listdir(img_path)[0]
        img = scipy.misc.imread(os.path.join(img_path, img_name))
        if (img.shape[0] + img.shape[1] != 512):
            print(img.shape)
            shutil.move(sample_path, save_path)

if __name__ == '__main__':
    # process_nuclei()
    move_files()









