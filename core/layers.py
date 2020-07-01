# -*- coding: utf-8 -*-
"""
Auxiliary functions and operations for network construction, some of which have
been deprecated for high-level modules in TensorFlow.

@author: Xinzhe Luo
"""

from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow as tf
import numpy as np


def weight_variable(shape, name="weight"):
    fan_in, fan_out = shape[-2:]
    low = -1*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation 
    high = 1*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32), name=name)


def weight_variable_devonc(shape, name="weight_deconv"):
    fan_in, fan_out = shape[-2:]
    low = -1*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation 
    high = 1*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32), name=name)


def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, keep_prob_):
    conv_2d = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.dropout(conv_2d, keep_prob_)


def deconv2d(x, w, stride):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, stride, stride, 1], padding='VALID')


def max_pool2d(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')


'''
def batch_norm(x, train_phase):
    x_norm = tf.layers.batch_normalization(x, axis=0, training=train_phase)
    return x_norm
'''


def batch_norm(x, name_scope, training, size, epsilon=1e-3, decay=0.999):
    """
    Assume 4d [batch_size, ny, nx, feature_size] tensor
    size = output feature size
    """

    with tf.variable_scope(name_scope):
        scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', [size])

        pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
        pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])

        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)


def crop_and_concat(x1, x2):
    """
    Crop x1 to match the size of x2 and concatenate them.
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1]-x2_shape[1])//2, (x1_shape[2]-x2_shape[2])//2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size, name='crop')
    crop_concat = tf.concat([x1_crop, x2], -1, name='crop_concat')
    crop_concat.set_shape([None, None, None, x1.get_shape().as_list()[-1] + x2.get_shape().as_list()[-1]])
    return crop_concat


def crop_and_add(x1, x2):
    """
    Crop x1 to match the size of x2 and add them together.
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1]-x2_shape[1])//2, (x1_shape[2]-x2_shape[2])//2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size, name='crop')
    return tf.add(x1_crop, x2, name='crop_add')


def pad_and_concat(x1, x2):
    """
    Pad x2 to match the size of x1 and concatenate them.
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    offsets = [0, (x1_shape[1]-x2_shape[1])//2, (x1_shape[2]-x2_shape[2])//2, 0]
    paddings = [[0, 0],
                [offsets[1], x1_shape[1]-x2_shape[1]-offsets[1]],
                [offsets[2], x1_shape[2]-x2_shape[2]-offsets[2]],
                [0, 0]]
    x2_pad = tf.pad(x2, paddings, name='pad')
    pad_concat = tf.concat([x1, x2_pad], -1, name='pad_concat')
    pad_concat.set_shape([None, None, None, x1.get_shape().as_list()[-1] + x2.get_shape().as_list()[-1]])
    return pad_concat


def pad_and_add(x1, x2):
    """
    Pad x2 to match the size of x1 and add them together.
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    offsets = [0, (x1_shape[1]-x2_shape[1])//2, (x1_shape[2]-x2_shape[2])//2, 0]
    paddings = [[0, 0],
                [offsets[1], x1_shape[1]-x2_shape[1]-offsets[1]],
                [offsets[2], x1_shape[2]-x2_shape[2]-offsets[2]],
                [0, 0]]
    x2_pad = tf.pad(x2, paddings, name='pad')
    return tf.add(x1, x2_pad, name='pad_add')


def crop_to_tensor(x1, x2):
    """
    Crop tensor x1 to match the shape of x2.
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1]-x2_shape[1])//2, (x1_shape[2]-x2_shape[2])//2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size, name='crop')
    x1_crop.set_shape([None, None, None, x1.get_shape().as_list()[-1]])
    return x1_crop


def crop_to_tensor_3d(x1, x2):
    """
    Crop tensor x1 to match the shape of x2.
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1]-x2_shape[1])//2, (x1_shape[2]-x2_shape[2])//2, (x1_shape[3]-x2_shape[3])//2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size, name='crop')
    x1_crop.set_shape([None, None, None, None, x1.get_shape().as_list()[-1]])
    return x1_crop


def pad_to_tensor(x1, x2):
    """
    Pad tensor x1 to match the shape of x2.
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    offsets = [0, (x2_shape[1] - x1_shape[1]) // 2, (x2_shape[2] - x1_shape[2]) // 2, 0]
    paddings = [[0, 0],
                [offsets[1], x2_shape[1] - x1_shape[1] - offsets[1]],
                [offsets[2], x2_shape[2] - x1_shape[2] - offsets[2]],
                [0, 0]]

    x1_pad = tf.pad(x1, paddings, name='pad')

    return x1_pad


def pixel_wise_softmax(output_map):
    """
    deprecated function for tf.nn.softmax
    """
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map, tf.reverse(exponential_map, [False, False, False, True]))
    return tf.divide(exponential_map, evidence, name="pixel_wise_softmax")


def pixel_wise_softmax_2(output_map):
    """
    deprecated function for tf.nn.softmax
    """
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, -1, keepdims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[-1]]))
    return tf.clip_by_value(tf.divide(exponential_map, tensor_sum_exp), 1e-10, 1.0)


def cross_entropy_map(labels, probs):
    """
    Compute the element-wise cross-entropy map by clipping the values of softmax probabilities to avoid Nan loss.

    :param labels: ground-truth value using one-hot representation
    :param probs: probability map as the output of softmax
    :return: A tensor of the same shape as lables and of the same shape as probs with the cross entropy loss.
    """
    return tf.reduce_sum(- labels * tf.log(tf.clip_by_value(probs, 1e-10, 1.0)), axis=-1, name="cross_entropy_map")


def balance_weight_map(flat_labels):
    """
    :param flat_labels: masked ground truth tensor in shape [-1, n_class]
    :return the balance weight map in 1-D tensor
    """
    n = tf.shape(flat_labels)[0]
    return tf.reduce_sum(tf.multiply(flat_labels, tf.tile(1 / tf.reduce_sum(flat_labels, axis=0, keepdims=True),
                                                          [n, 1])), axis=-1, name='balance_weight_map')


def gaussian_noise_layer(input_layer, std):
    """
    Apply Gaussian noise to the input.

    :param input_layer: Inputs.
    :param std: Standard deviation for the noise.
    :return: Blurred features.
    """
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def linear_additive_upsample(input_tensor, new_size=2, n_split=4):
    """
    Apply linear additive up-sampling layer, described in paper Wojna et al., The devil is in the decoder,
        https://arxiv.org/abs/1707.05847.

    :param input_tensor: Input tensor.
    :param new_size: The factor of up-sampling.
    :param n_split: The n_split consecutive channels are added together.
    :return: Linearly additively upsampled feature maps.
    """
    with tf.name_scope('linear_additive_upsample'):
        n_channels = input_tensor.get_shape().as_list()[-1]
        input_dim = input_tensor.get_shape().ndims

        assert n_split > 0 and n_channels % n_split == 0, "Number of feature channels should be divisible by n_splits."

        if input_dim == 4:
            upsample = tf.keras.layers.UpSampling2D(size=new_size, name='upsample')(input_tensor)
        elif input_dim == 5:
            upsample = tf.keras.layers.UpSampling3D(size=new_size, name='upsample')(input_tensor)
        else:
            raise TypeError('Incompatible input spatial rank: %d' % input_dim)

        split = tf.split(upsample, n_split, axis=-1)
        split_tensor = tf.stack(split, axis=-1)
        output_tensor = tf.reduce_sum(split_tensor, axis=-1, name='output_tensor')

    return output_tensor


def residual_additive_upsample(inputs, filter_size, strides, feature_size, n_split, regularizer, train_pahse, trainable,
                               name_or_scope='residual_additive_upsample'):
    n_channel = inputs.get_shape().as_list()[-1]
    assert n_channel == feature_size * n_split, "The number of input channels must be the product of output feature " \
                                                "size and the number of splits."
    with tf.variable_scope(name_or_scope):
        deconv = tf.layers.conv2d_transpose(inputs, filters=feature_size, kernel_size=filter_size, strides=strides,
                                            padding='same', use_bias=False,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            kernel_regularizer=regularizer, trainable=trainable, name='deconv')
        bn = tf.layers.batch_normalization(deconv, training=train_pahse, trainable=trainable, name='bn')
        relu = tf.nn.relu(bn, name='relu')
        upsample = linear_additive_upsample(inputs, strides, n_split)

        return tf.add(relu, upsample, name='res_upsample')

