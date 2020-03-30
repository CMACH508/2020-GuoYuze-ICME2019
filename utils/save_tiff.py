import os
import scipy.misc
import imageio
from libtiff import TIFF

def tiff_to_image_array(tiff_image_name, out_folder, out_type):

    if os.path.exists(out_folder) is False:
        os.makedirs(out_folder)

    tif = TIFF.open(tiff_image_name, mode='r')
    idx = 0
    for im in list(tif.iter_images()):
        im_name = out_folder + '/' + str(idx) + out_type
        imageio.imsave(im_name, im)
        print("successfully saved!")
        idx +=1

    return

def image_array_to_tiff(tiff_image_name, in_folder):
    tif = TIFF.open(os.path.join(in_folder, tiff_image_name), mode='w')
    for i in range(30):
        img = imageio.imread(os.path.join(in_folder, '%d-label.png' % i))
        tif.write_image(img)

def transfer_tif(in_path, out_path):
    in_tif = TIFF.open(in_path, mode='r')
    out_tif = TIFF.open(out_path, mode='w')

    for im in list(in_tif.iter_images()):
        im = 1 - im
        out_tif.write_image(im)

if __name__ == '__main__':
    image_array_to_tiff('results.tiff', './results/lmser/ISBI2012_R/')
