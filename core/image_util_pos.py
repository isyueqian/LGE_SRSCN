"""
author: Qian Yue
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import glob
import re
import numpy as np
from PIL import Image
import nibabel as nib
import tensorflow as tf


class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    """

    channels = 1
    n_class = 4

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label, affine, position = self._next_data()

        train_data = self._process_data(data)
        if self.inference_phase:
            labels = self._process_labels_3d(label)
        else:
            labels = self._process_labels_2d(label)

        train_data, labels = self._post_process(train_data, labels)

        nx = train_data.shape[0]
        ny = train_data.shape[1]
        nz = train_data.shape[2]
        if self.inference_phase:
            return train_data.reshape(1, nx, ny, nz, self.channels), labels.reshape(1, nx, ny, nz, self.n_class), affine, position
        else:
            return train_data.reshape(1, nx, ny, self.channels), labels.reshape(1, nx, ny, self.n_class), affine, position

    def _process_labels_2d(self, label):
        """
        processed labels for 2d training.
        :param label: expected dim [height, width, 1]
        :return: one-hot labels with dim [height, width, n_class]
        """
        label = np.squeeze(label, axis=-1)
        nx = label.shape[0]
        ny = label.shape[1]

        labels = np.zeros((nx, ny, self.n_class), dtype=np.float32)

        for k in range(self.n_class):
            labels[..., k][label == self.label_intensity[k]] = 1
        return labels

    def _process_labels_3d(self, label):
        """
        processed labels for 3d inference.
        :param label: expected dim [height, width, slices]
        :return: one-hot labels with dim [height, width, slices, n_class]
        """
        nx = label.shape[0]
        ny = label.shape[1]
        nz = label.shape[2]
        labels = np.zeros((nx, ny, nz, self.n_class), dtype=np.float32)

        for k in range(self.n_class):
            labels[..., k][label == self.label_intensity[k]] = 1

        return labels

    def _process_data(self, data):
        """
        processed data for both situation.
        :param data: epected dim [height, width, 1] or [height, width, slices]
        :return: normalized data with the same dimensions.
        """
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        '''
        # max-min normalization
        data -= np.amin(data)
        data /= np.amax(data)
        '''
        # z-score normalization
        eps = 1e-5
        data = (data - np.mean(data)) / (np.std(data)+eps)
        return data

    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        """
        '''
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        concat_image = tf.concat([tf.expand_dims(data, -1), tf.expand_dims(labels, -1)], axis=-1)

        maybe_flipped = tf.image.random_flip_left_right(concat_image)
        maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)

        data = maybe_flipped[..., :1]
        labels = maybe_flipped[..., 1:]

        data = tf.image.random_brightness(data, 0.2)
        # labels = tf.image.random_brightness(labels, 0.7)
        '''
        return data, labels

    def __call__(self, n):
        if self.crop_patch:
            train_data, labels, affine, position = self._load_data_and_label()

            nx = train_data.shape[1]
            ny = train_data.shape[2]
            nz = train_data.shape[3]

            X = np.zeros((n, nx, ny, self.channels))
            Y = np.zeros((n, nx, ny, self.n_class))

            Z = [affine]
            P = [position]

            X[0] = train_data
            Y[0] = labels
            for i in range(1, n):
                train_data, labels, affine, position = self._load_data_and_label()
                X[i] = train_data
                Y[i] = labels
                Z.append(affine)
                P.append(position)
        else:
            X = []
            Y = []
            Z = []
            P = []
            for _ in range(n):
                train_data, labels, affine, position = self._load_data_and_label()
                train_data = np.squeeze(train_data, axis=0).transpose((2, 0, 1, 3))
                labels = np.squeeze(labels, axis=0).transpose((2, 0, 1, 3))
                X.append(train_data)
                Y.append(labels)
                Z.append(affine)
                P.append(position)

            assert len(X[0].shape) == 4 and len(Y[0].shape) == 4, "Not the right dimension for input data!"

        return X, Y, Z, P


class SimpleDataProvider(BaseDataProvider):
    """
    A simple data provider for numpy arrays. 
    Assumes that the data and label are numpy array with the dimensions
    data `[n, X, Y, channels]`, label `[n, X, Y, classes]`. Where
    `n` is the number of images, `X`, `Y` the size of the image.
    :param data: data numpy array. Shape=[n, X, Y, channels]
    :param label: label numpy array. Shape=[n, X, Y, classes]
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """

    def __init__(self, data, label, a_min=None, a_max=None, channels=1, n_class=2):
        super(SimpleDataProvider, self).__init__(a_min, a_max)
        self.data = data
        self.label = label
        self.file_count = data.shape[0]
        self.n_class = n_class
        self.channels = channels

    def _next_data(self):
        idx = np.random.choice(self.file_count)
        return self.data[idx], self.label[idx]


class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file prefix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_label.tif'
    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param label_suffix: suffix pattern for the label images. Default '_label.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param crop_patch: if patches of a certain size need to be cropped for training. Default 'True'
    :param patch_size: size of the patch. Default '(64, 64, 64)', set -1 for axes not to be cropped
    :param center_roi: roi size
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    :param contain_foreground: if the patch should contain foreground, default=False.
    :param label_intensity: list of intensities of the ground truth
    
    """

    def __init__(self, search_path, inference_phase, a_min=None, a_max=None, data_suffix=".tif", label_suffix='_label.tif',
                 shuffle_data=True, crop_patch=True, center_crop=False, patch_size=(64, 64, 64), center_roi=(80, 80, 1), channels=3, n_class=4,
                 contain_foreground=False, label_intensity=(0, 420)):
        super(ImageDataProvider, self).__init__(a_min, a_max)

        self.data_suffix = data_suffix
        self.label_suffix = label_suffix
        self.file_index = -1
        self.shuffle_data = shuffle_data
        self.crop_patch = crop_patch
        self.center_crop = center_crop
        self.patch_size = patch_size
        self.center_roi = center_roi
        self.n_class = n_class
        self.channels = channels
        self.contain_foreground = contain_foreground
        self.label_intensity = label_intensity
        self.inference_phase = inference_phase

        self.data_files = self._find_data_files(search_path)

        if self.shuffle_data:
            np.random.shuffle(self.data_files)

        assert len(self.data_files) > 0, "No training files"
        assert len(self.label_intensity) == self.n_class, "Number of label intensities don't epual to number of classes"
        print("Number of files used: %s" % len(self.data_files))
    
    def reset_index(self):
        self.file_index = -1
    
    def _find_data_files(self, search_path):
        all_files = strsort(glob.glob(search_path))
        return [name for name in all_files if self.data_suffix in name and self.label_suffix not in name]

    def _load_file_2d(self, path, dtype=np.float32):
        """
        load files from raw type .png
        read .png, get shape [height, width]
        :param path: read from this path
        :param dtype: set the data type of object
        :return: ndarray of data
        """

        img = Image.open(path)
        img_array = np.array(img, dtype)
        return img_array.transpose((1, 0))

    def _load_file_3d(self, path, dtype=np.float32):
        """
        read .nii.gz, get shape [height, width, slices]
        :param path: read from this path
        :param dtype: set the data type of object
        :return: ndarray of data and its affine matrix
        """
        img = nib.load(path)
        img_array = img.get_fdata(dtype=dtype)
        if 'lab' in path:
            img_array[:][img_array == 200] = 200
            img_array[:][img_array == 500] = 244
            img_array[:][img_array == 600] = 88
        # print(np.unique(img_array))
        return img_array.transpose((1, 0, 2)), img.affine

    def _cycle_file(self):
        self.file_index += 1
        if self.file_index >= len(self.data_files):
            self.file_index = 0
            if self.shuffle_data:
                np.random.shuffle(self.data_files)
    
    def _next_data(self):
        """
        processed the data from dimension view
        :return: if center_crop, return [roi_height, roi_width, 1] or [roi_height, roi_width, slices]
        """
        self._cycle_file()

        image_name = self.data_files[self.file_index]
        label_name = image_name.replace(self.data_suffix, self.label_suffix)

        if self.inference_phase:
            label, affine = self._load_file_3d(label_name, np.float32)
            image, _ = self._load_file_3d(image_name, np.float32)
            position = None
        else:
            label = self._load_file_2d(label_name, np.float32)
            image = self._load_file_2d(image_name, np.float32)
            label = np.expand_dims(label, axis=-1)
            image = np.expand_dims(image, axis=-1)
            affine = None
            position = np.float32(label_name.split('_')[3])

        if self.center_crop:
            assert np.all(np.array(self.center_roi) <= np.array(label.shape)), print(
                'Patch size exceeds dimensions.')
            center = compute_center(label)
            where_are_nan = np.isnan(center)
            center[where_are_nan] = int(label.shape[0] // 2)

            x = np.array([center[i][0] for i in range(label.shape[-1])]).astype(np.int)
            y = np.array([center[i][1] for i in range(label.shape[-1])]).astype(np.int)

            x = x[0]
            y = y[0]

            beginx = x - self.center_roi[0]
            beginy = y - self.center_roi[1]
            endx = x + self.center_roi[0]
            endy = y + self.center_roi[1]

            gt = label[beginx:endx, beginy:endy, :]
            img = image[beginx:endx, beginy:endy, :]

        elif self.crop_patch:

            assert np.all(np.array(self.patch_size) <= np.array(label.shape)), print('Patch size exceeds dimensions.')

            x = np.random.randint(self.patch_size[0] // 2,
                                  label.shape[0] + self.patch_size[0] // 2 - self.patch_size[0] + 1)
            y = np.random.randint(self.patch_size[1] // 2,
                                  label.shape[1] + self.patch_size[1] // 2 - self.patch_size[1] + 1)
            z = np.random.randint(self.patch_size[2] // 2,
                                  label.shape[2] + self.patch_size[2] // 2 - self.patch_size[2] + 1)

            begin = np.where(np.equal(self.patch_size, -1), 0, [x - self.patch_size[0] // 2,
                                                                y - self.patch_size[1] // 2,
                                                                z - self.patch_size[2] // 2])

            end = np.where(np.equal(self.patch_size, -1), [label.shape[0],
                                                           label.shape[1],
                                                           label.shape[2]],
                           [x + self.patch_size[0] - self.patch_size[0] // 2,
                            y + self.patch_size[1] - self.patch_size[1] // 2,
                            z + self.patch_size[2] - self.patch_size[2] // 2])

            gt = label[begin[0]:end[0], begin[1]:end[1], begin[2]:end[2]]
            img = image[begin[0]:end[0], begin[1]:end[1], begin[2]:end[2]]
        else:
            gt = label
            img = image

        if self.contain_foreground:

            while not np.any([gt == k for k in self.label_intensity[1:]]):
                self._cycle_file()

                image_name = self.data_files[self.file_index]
                label_name = image_name.replace(self.data_suffix, self.label_suffix)

                if self.inference_phase:
                    label, affine = self._load_file_3d(label_name, np.float32)
                    image, _ = self._load_file_3d(image_name, np.float32)
                else:
                    label = self._load_file_2d(label_name, np.float32)
                    image = self._load_file_2d(image_name, np.float32)
                    label = np.expand_dims(label, axis=-1)
                    image = np.expand_dims(image, axis=-1)

                if self.center_crop:
                    assert np.all(np.array(self.center_roi) <= np.array(label.shape)), print(
                        'Patch size exceeds dimensions.')
                    center = compute_center(label)
                    where_are_nan = np.isnan(center)
                    center[where_are_nan] = int(label.shape[0] // 2)

                    x = np.array([center[i][0] for i in range(label.shape[-1])]).astype(np.int)
                    y = np.array([center[i][1] for i in range(label.shape[-1])]).astype(np.int)

                    x = x[0]
                    y = y[0]

                    beginx = x - self.center_roi[0]
                    beginy = y - self.center_roi[1]
                    endx = x + self.center_roi[0]
                    endy = y + self.center_roi[1]

                    gt = label[beginx:endx, beginy:endy, :]
                    img = image[beginx:endx, beginy:endy, :]

                elif self.crop_patch:

                    assert np.all(np.array(self.patch_size) <= np.array(label.shape)), print(
                        'Patch size exceeds dimensions.')
                    x = np.random.randint(self.patch_size[0] // 2,
                                          label.shape[0] + self.patch_size[0] // 2 - self.patch_size[0] + 1)
                    y = np.random.randint(self.patch_size[1] // 2,
                                          label.shape[1] + self.patch_size[1] // 2 - self.patch_size[1] + 1)
                    z = np.random.randint(self.patch_size[2] // 2,
                                          label.shape[2] + self.patch_size[2] // 2 - self.patch_size[2] + 1)
                    begin = np.where(np.equal(self.patch_size, -1), 0, [x - self.patch_size[0] // 2,
                                                                        y - self.patch_size[1] // 2,
                                                                        z - self.patch_size[2] // 2])
                    end = np.where(np.equal(self.patch_size, -1), [label.shape[0],
                                                                   label.shape[1],
                                                                   label.shape[2]],
                                   [x + self.patch_size[0] - self.patch_size[0] // 2,
                                    y + self.patch_size[1] - self.patch_size[1] // 2,
                                    z + self.patch_size[2] - self.patch_size[2] // 2])

                    position = np.arange(begin[2], end[2]) / label.shape[2]

                    gt = label[begin[0]:end[0], begin[1]:end[1], begin[2]:end[2]]
                    img = image[begin[0]:end[0], begin[1]:end[1], begin[2]:end[2]]
                else:
                    gt = label
                    img = image

        return img, gt, affine, position


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    """
    return [atoi(c) for c in re.split('(\d+)', text)]


def strsort(alist):
    alist.sort(key=natural_keys)
    return alist


def compute_center(label):
    points = np.where(label > 0)
    return np.array([[np.average(points[0][points[2] == j]), np.average(points[1][points[2] == j])] for j in range(label.shape[-1])])
