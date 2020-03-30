import scipy.misc
from libtiff import TIFF
import os 

def tiff_to_image_array(tiff_image_name, out_folder, out_type):

    if os.path.exists(out_folder) is False:
        os.makedirs(out_folder)

    tif = TIFF.open(tiff_image_name, mode='r')
    idx = 0
    for im in list(tif.iter_images()):
        im_name = out_folder + '/' + str(idx) + out_type
        scipy.misc.imsave(im_name, im)
        print("successfully saved!")
        idx +=1

    return 

if __name__ == '__main__':
    tiff_to_image_array('./train-volume.tif', './train_img', '.png')
    tiff_to_image_array('./train-labels.tif', './train_label', '.png')
    tiff_to_image_array('./test-volume.tif', './test_img_1', '.png')
