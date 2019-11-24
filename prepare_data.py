"""
This script is used to prepare the dataset for training and test.

Author: Hui Qu
"""


import os
import numpy as np
import math
from skimage import morphology, io, color


def main():
    original_data_dir = './original_data/MultiOrgan'

    img_dir = '{:s}/images'.format(original_data_dir)
    label_instance_dir = '{:s}/labels_instance'.format(original_data_dir)
    label_dir = '{:s}/labels'.format(original_data_dir)
    weightmap_dir = '{:s}/weight_maps'.format(original_data_dir)
    patch_folder = '{:s}/patches'.format(original_data_dir)

    create_folder(patch_folder)

    # ------ color normalization for all images
    anchor_img_path = './original_data/MultiOrgan/ori_images/Prostate_TCGA-G9-6356-01Z-00-DX1.tif'
    color_norm(anchor_img_path, '{:s}/original_images'.format(original_data_dir), img_dir)

    # ------ create ternary labels from instance labels
    create_ternary_labels(label_instance_dir, label_dir)

    # ------ create weight maps from instance labels
    # use matlab code weight_map.m for parallel computing

    # ------ split large images into 250x250 patches
    print("Splitting large images into small patches...")
    split_patches(img_dir, '{:s}/images'.format(patch_folder))
    split_patches(label_dir, '{:s}/labels'.format(patch_folder), 'label')
    split_patches(weightmap_dir, '{:s}/weight_maps'.format(patch_folder), 'weight')


def create_ternary_labels(data_dir, save_dir):
    """ create ternary labels from instance labels """

    create_folder(save_dir)

    print("Generating ternary labels from instance labels...")
    image_list = os.listdir(data_dir)
    for image_name in sorted(image_list):
        name = image_name.split('.')[0]

        image_path = os.path.join(data_dir, image_name)
        image = io.imread(image_path)
        h, w = image.shape

        # extract edges
        id_max = np.max(image)
        contours = np.zeros((h, w), dtype=np.bool)
        nuclei_inside = np.zeros((h, w), dtype=np.bool)
        for i in range(1, id_max+1):
            nucleus = image == i
            nuclei_inside += morphology.erosion(nucleus)
            contours += morphology.dilation(nucleus) & (~morphology.erosion(nucleus))

        ternary_label = np.zeros((h, w, 3), np.uint8)
        ternary_label[:, :, 0] = nuclei_inside.astype(np.uint8) * 255       # inside
        ternary_label[:, :, 1] = contours.astype(np.uint8) * 255            # contours
        ternary_label[:, :, 2] = (~(nuclei_inside + contours)).astype(np.uint8) * 255  # background

        io.imsave('{:s}/{:s}_label.png'.format(save_dir, name), ternary_label.astype(np.uint8))


def split_patches(data_dir, save_dir, postfix=None):
    """ split large image into small patches """
    create_folder(save_dir)

    image_list = os.listdir(data_dir)
    for image_name in image_list:
        name = image_name.split('.')[0]
        if postfix and name[-len(postfix):] != postfix:
            continue
        image_path = os.path.join(data_dir, image_name)
        image = io.imread(image_path)
        seg_imgs = []

        # split into 16 patches of size 250x250
        h, w = image.shape[0], image.shape[1]
        patch_size = 250
        h_overlap = math.ceil((4 * patch_size - h) / 3)
        w_overlap = math.ceil((4 * patch_size - w) / 3)
        for i in range(0, h-patch_size+1, patch_size-h_overlap):
            for j in range(0, w-patch_size+1, patch_size-w_overlap):
                if len(image.shape) == 3:
                    patch = image[i:i+patch_size, j:j+patch_size, :]
                else:
                    patch = image[i:i + patch_size, j:j + patch_size]
                seg_imgs.append(patch)

        for k in range(len(seg_imgs)):
            if postfix:
                io.imsave('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(postfix)-1], k, postfix), seg_imgs[k])
            else:
                io.imsave('{:s}/{:s}_{:d}.png'.format(save_dir, name, k), seg_imgs[k])


def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def color_norm(anchor_img_path, data_dir, save_dir):
    create_folder(save_dir)

    anchor_img = io.imread(anchor_img_path)
    normalizer = Reinhard_normalizer()

    normalizer.fit(anchor_img)

    file_list = os.listdir(data_dir)
    for file in sorted(file_list):
        name = file.split('.')[0]
        img = io.imread('{:s}/{:s}'.format(data_dir, file))
        normalized_img = normalizer.transform(img)
        io.imsave('{:s}/{:s}.png'.format(save_dir, name), normalized_img)


class Reinhard_normalizer(object):
    """
    color normalization using Reinhard's method
    """

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target):
        # target = self._standardize_brightness(target)
        means, stds = self._get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def transform(self, I):
        # I = self._standardize_brightness(I)
        I1, I2, I3 = self._lab_split(I)
        means, stds = self._get_mean_std(I)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return self._merge_back(norm1, norm2, norm3)

    def _lab_split(self, I):
        """
        Convert from RGB uint8 to LAB and split into channels
        """
        I = color.rgb2lab(I)
        I1, I2, I3 = I[:,:,0], I[:,:,1], I[:,:,2]

        return I1, I2, I3

    def _merge_back(self, I1, I2, I3):
        """
        Take seperate LAB channels and merge back to give RGB uint8
        """
        I = np.stack((I1, I2, I3), axis=2)
        return (color.lab2rgb(I) * 255).astype(np.uint8)

    def _get_mean_std(self, I):
        """
        Get mean and standard deviation of each channel
        """
        I1, I2, I3 = self._lab_split(I)
        means = np.mean(I1), np.mean(I2), np.mean(I3)
        stds = np.std(I1), np.std(I2), np.std(I3)
        return means, stds

    def _standardize_brightness(self, I):
        p = np.percentile(I, 90)
        return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    main()
