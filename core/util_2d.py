# -*- coding: utf-8 -*-
"""
Functions and operations for performance visualization and result store,
some of which are not used in the current situation.

@author: Qian Yue
"""
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import os
import logging
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from core import image_util


def plot_prediction(x_test, y_test, prediction, save=False):
    test_size = x_test.shape[0]
    fig, ax = plt.subplots(test_size, 3, figsize=(12, 12), sharey=True, sharex=True)

    x_test = crop_to_shape(x_test, prediction.shape)
    y_test = crop_to_shape(y_test, prediction.shape)

    ax = np.atleast_2d(ax)
    for i in range(test_size):
        cax = ax[i, 0].imshow(x_test[i])
        plt.colorbar(cax, ax=ax[i, 0])
        cax = ax[i, 1].imshow(y_test[i, ..., 1])
        plt.colorbar(cax, ax=ax[i, 1])
        pred = prediction[i, ..., 1]
        pred -= np.amin(pred)
        pred /= np.amax(pred)
        cax = ax[i, 2].imshow(pred)
        plt.colorbar(cax, ax=ax[i, 2])
        if i == 0:
            ax[i, 0].set_title("x")
            ax[i, 1].set_title("y")
            ax[i, 2].set_title("pred")
    fig.tight_layout()

    if save:
        fig.savefig(save)
    else:
        fig.show()
        plt.show()


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [n, nx, ny, channels]
    
    :returns img: the rgb image [n, nx, ny, 3]
    """
    if len(img.shape) < 4:
        img = np.expand_dims(img, axis=-1)
    channels = img.shape[-1]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    for k in range(np.shape(img)[3]):
        st = img[:, :, :, k]
        if np.amin(st) != np.amax(st):
            st -= np.amin(st)
            st /= np.amax(st)
        st *= 255
    return img


'''
def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border 
    (expects a tensor of shape [batches, nx, ny, channels].
    
    :param data: the array to crop, shape=[n, nx, ny, nz, n_class]
    :param shape: the target shape
    """
    assert np.all(data.shape>=shape), "The shape of array to be cropped is smaller than the target shape."
    offset1 = (data.shape[1] - shape[1])//2
    offset2 = (data.shape[2] - shape[2])//2
    offset3 = (data.shape[3] - shape[3])//2
    
    return data[:, offset1:(offset1+shape[1]), offset2:(offset2+shape[2]), offset3:(offset3+shape[3]), :]
'''


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border
    (expects a tensor of shape [batches, nx, ny, channels]).

    :param data: the array to crop, shape=[n, nx, ny, n_class/channels]
    :param shape: the target shape
    """
    assert np.all(data.shape[1:3] >= shape[1:3]), "The shape of array to be cropped is smaller than the target shape."
    offset0 = (data.shape[1] - shape[1]) // 2
    offset1 = (data.shape[2] - shape[2]) // 2
    remainder0 = (data.shape[1] - shape[1]) % 2
    remainder1 = (data.shape[2] - shape[2]) % 2

    if (data.shape[1] - shape[1]) == 0 and (data.shape[2] - shape[2]) == 0:
        return data

    elif (data.shape[1] - shape[1]) != 0 and (data.shape[2] - shape[2]) == 0:
        return data[:, offset0:(-offset0 - remainder0), ]

    elif (data.shape[1] - shape[1]) == 0 and (data.shape[2] - shape[2]) != 0:
        return data[:, :, offset1:(-offset1 - remainder1), ]

    elif (data.shape[1] - shape[1]) != 0 and (data.shape[2] - shape[2]) != 0:
        return data[:, offset0:(-offset0 - remainder0), offset1:(-offset1 - remainder1), ]

    else:
        return data[:, offset0:(-offset0 - remainder0), offset1:(-offset1 - remainder1), ]


def pad_to_shape(data, shape):
    """
    Pad the array to the given shape by the edge values
    """
    assert np.all(data.shape <= shape), "The shape of array to be padded is larger than the target shape."
    offset1 = (shape[1] - data.shape[1]) // 2
    offset2 = (shape[2] - data.shape[2]) // 2
    offset3 = (shape[3] - data.shape[3]) // 2
    remainder1 = (shape[1] - data.shape[1]) % 2
    remainder2 = (shape[2] - data.shape[2]) % 2
    remainder3 = (shape[3] - data.shape[3]) % 2

    return np.pad(data, (
        (0, 0), (offset1, offset1 + remainder1), (offset2, offset2 + remainder2), (offset3, offset3 + remainder3),
        (0, 0)),
                  'edge')


def combine_img_prediction(data, gt, pred):
    """
    Combines the data, grouth thruth and the prediction into one rgb image for each class
    
    :param data: the data tensor
    :param gt: the ground truth tensor
    :param pred: the prediction tensor
    
    :returns img: the concatenated rgb image 
    """
    ny = pred.shape[2]
    imgs = []
    for k in range(gt.shape[-1]):
        img = np.concatenate(
            (to_rgb(crop_to_shape(data, pred.shape)).reshape(-1, ny, 3, order='F'),
             to_rgb(crop_to_shape(gt[..., k], pred.shape)).reshape(-1, ny, 3, order='F'),
             to_rgb(pred[..., k]).reshape(-1, ny, 3, order='F')), axis=1)
        imgs.append(img)
    return imgs


def save_image(imgs, path):
    """
    Writes the image to disk
    
    :param imgs: the rgb images to save
    :param path: the target path
    """
    for i in range(len(imgs)):
        img_path = os.path.join(os.path.split(path)[0], 'class%d_' % i + os.path.split(path)[1])
        Image.fromarray(imgs[i].round().astype(np.uint8)).save(img_path, 'PNG', dpi=[300, 300], quality=95)


def save_prediction(data, gt, predictions, path):
    """
    Combine each prediction and the corresponding ground truth as well as input into one image and save as png files.

    :param data: list of input raw images
    :param gt: list of ground truth images
    :param predictions: list of predictions
    :param path: has the form of 'directory'
    """
    assert len(data) == len(gt) and len(gt) == len(predictions), print('Numbers of images are not equal.')
    abs_pred_path = os.path.abspath(path)
    if not os.path.exists(abs_pred_path):
        logging.info("Allocating '{:}'".format(abs_pred_path))
        os.makedirs(abs_pred_path)

    for k in range(len(data)):
        pred = np.where(np.equal(np.max(predictions[k], -1, keepdims=True), predictions[k]),
                        np.ones_like(predictions[k]),
                        np.zeros_like(predictions[k]))
        save_image(combine_img_prediction(data[k], gt[k], pred), os.path.join(path, 'sub%s.png' % k))


def save_prediction_1(predictions, affine, path):
    """
    Save the predictions into nibabel images.
    Predictions are pre-processed by setting the maximum along classes to be a certain intensity specific to its class.

    :param predictions: list of predictions
    :param affine: list of corresponding coordinates
    :param path: has the form of 'directory'
    """
    abs_pred_path = os.path.abspath(path)
    if not os.path.exists(abs_pred_path):
        logging.info("Allocating '{:}'".format(abs_pred_path))
        os.makedirs(abs_pred_path)

    for k in range(len(predictions)):
        pred = np.squeeze(predictions[k])
        intensity = np.tile(np.arange(0, 1000, 999 // (pred.shape[-1] - 1)), np.concatenate((pred.shape[:-1], [1])))
        mask = np.equal(np.max(pred, -1, keepdims=True), pred)
        img = nib.Nifti1Image(np.sum(mask * intensity, axis=-1).astype(np.float32).transpose((1, 0, 2)),
                              affine=affine[k])
        nib.save(img, os.path.join(path, 'sub%s.nii.gz' % k))


def save_prediction_2(predictions, path):
    """
    Save the predictions into numpy array.

    :param predictions: list of predictions
    :param path: has the form of 'directory'
    """
    abs_pred_path = os.path.abspath(path)
    if not os.path.exists(abs_pred_path):
        logging.info("Allocating '{:}'".format(abs_pred_path))
        os.makedirs(abs_pred_path)

    for k in range(len(predictions)):
        pred = predictions[k]
        np.save(os.path.join(path, 'sub%s.npy' % k), pred)


def plot_acc_auc_sens_spec(train_acc, train_auc, train_sens, train_spec, save_path):
    t = range(1, len(train_acc) + 1)
    plt.figure(1)
    plt.subplot(221)
    plt.plot(t, train_acc, label=save_path)
    plt.ylim([0.0, 1.05])
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.subplot(222)
    plt.plot(t, train_auc, label=save_path)
    plt.ylim([0.0, 1.05])
    plt.xlabel('Training Epochs')
    plt.ylabel('AUC')
    plt.subplot(223)
    plt.plot(t, train_sens, label=save_path)
    plt.ylim([0.0, 1.05])
    plt.xlabel('Training Epochs')
    plt.ylabel('Sensitivity')
    plt.subplot(224)
    plt.plot(t, train_spec, label=save_path)
    plt.ylim([0.0, 1.05])
    plt.xlabel('Training Epochs')
    plt.ylabel('Specificity')
    plt.legend()
    plt.savefig(save_path, dpi=600)


"""
def plot_dice_coefficient(net, N, model_path_prefix, save_path):
    test_data, test_label = image_util.ImageDataProvider(search_path="test/*.npy",
                                                         data_prefix="org",
                                                         label_prefix="scr",
                                                         channels=1,
                                                         n_class=3,
                                                         shuffle_data=False)(20)

    path = './' + model_path_prefix + '(0)\model.cpkt'
    prediction = [net.predict(path, data) for data in test_data]
    for i in range(1, N):
        path = './' + model_path_prefix + '(%s)\model.cpkt' % i
        prediction += [net.predict(path, data) for data in test_data]
    ground_truth = []
    for label in test_label:
        ground_truth.append(crop_to_shape(label, prediction[test_label.index(label)].shape))
    # mask = np.tile(crop_to_shape(test_mask, pred.shape), [N, 1, 1, 1])[..., 1]
    t = np.arange(0, 1.01, 0.01)
    dc_mean = np.zeros_like(t)
    for k in range(0, 101):
        prediction_1 = [np.copy(pred) for pred in prediction]
        if k == 0:
            for pred_1 in prediction_1:
                pred_1[pred_1 >= t[k]] = 1
        else:
            for pred_1 in prediction_1:
                pred_1[pred_1 >= t[k]] = 1
                pred_1[pred_1 < t[k]] = 0
        dc_mean[k] = np.mean([2 * np.sum(pred_1 * ground_truth[prediction_1.index(pred_1)]) /
                              np.sum(pred_1 + ground_truth[prediction_1.index(pred_1)]) for pred_1 in prediction_1])
    plt.figure(2)
    plt.plot(t, dc_mean, label=save_path)
    plt.ylim([0.0, 1.0])
    plt.xlabel('Threshold')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.savefig(save_path + '.png', dpi=600)
    t_max = t[np.argmax(dc_mean)]
    dc = np.zeros(len(prediction))
    for k in range(len(prediction)):
        pred_2 = np.copy(prediction[k])
        gt_2 = np.copy(ground_truth[k])
        if t_max == 0:
            pred_2[pred_2 >= t_max] = 1
        else:
            pred_2[pred_2 >= t_max] = 1
            pred_2[pred_2 < t_max] = 0
        dc[k] = (2 * np.sum(pred_2 * gt_2)) / np.sum(pred_2 + gt_2)
    np.save(save_path + '.npy', dc)


def plot_roc_curve(net, N, model_path_prefix, save_path):
    test_data, test_label = image_util.ImageDataProvider(search_path="test/*.npy",
                                                         data_prefix="org",
                                                         label_prefix="scr",
                                                         channels=1,
                                                         shuffle_data=False)(20)

    path = './' + model_path_prefix + '(0)\model.cpkt'
    prediction = [net.predict(path, data) for data in test_data]
    for i in range(1, N):
        path = './' + model_path_prefix + '(%s)\model.cpkt' % i
        prediction += [net.predict(path, data) for data in test_data]
    ground_truth = []
    for label in test_label:
        ground_truth.append(crop_to_shape(label, prediction[test_label.index(label)].shape))
    pred = np.array([])
    for pred_1 in prediction:
        pred = np.hstack((pred, np.reshape(pred_1, [-1])))
    gt = np.array([])
    for gt_1 in ground_truth:
        gt = np.hstack((gt, np.reshape(gt_1, [-1])))
    fpr, tpr, thresholds = roc_curve(np.reshape(gt, [-1]), np.reshape(pred, [-1]), pos_label=1)
    roc_auc = auc(fpr, tpr)
    np.save(save_path, roc_auc)
    plt.figure(3)
    plt.plot(fpr, tpr, label='AUC= %.3f' % roc_auc)
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(save_path + '.png', dpi=600)


def compare_stats(stat_1, stat_2):
    from scipy import stats
    n = len(stat_1)
    T = np.sqrt(n) * np.mean(stat_1 - stat_2) / np.std(stat_1 - stat_2)
    pval = stats.t.sf(np.abs(T), n - 1) * 2
    t0 = stats.t.ppf(0.025, n - 1)
    if T > -t0:
        print('stat_1 is significantly greater than stat_2.')
    elif T < t0:
        print('stat_2 is significantly greater than stat_1.')
    else:
        print('no significant difference between two statistics.')
    print('t-statistic= %.3f p-value= %.4f' % (T, pval))
"""
