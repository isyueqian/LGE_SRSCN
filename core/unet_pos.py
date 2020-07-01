# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 2019

@author: Xinzhe Luo
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging
from sklearn.metrics import roc_auc_score

import tensorflow as tf

from core import util
from core.layers import *
from core.ACNN import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def create_position_branch(in_node, batch_size):
    in_node_flat = tf.reshape(in_node, [batch_size, 15*15*256*2], name='flat_node_fc')

    pos_fc_1 = tf.layers.dense(in_node_flat, units=256,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='position_fc_1')
    pos_fc_2 = tf.layers.dense(pos_fc_1, units=batch_size,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='position_fc_2')

    return tf.nn.sigmoid(pos_fc_2)


def create_conv_net(x, train_phase, dropout_rate=0.2, n_class=4,  regularizer=None, layers=5,
                    features_root=16, filter_size=3, pool_size=2, summaries=True, batch_size=32):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param dropout_rate: dropout probability tensor
    :param n_class: number of output labels
    :param train_phase: flag True for training and False for inference
    :param regularizer: Type of regularizer applied to the kernel weights.
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, "
                 "pool size: {pool_size}x{pool_size}".format(layers=layers, features=features_root,
                                                             filter_size=filter_size,
                                                             pool_size=pool_size))

    in_node = tf.cond(train_phase, lambda: gaussian_noise_layer(x, std=1.),
                      lambda: x, name='gaussian_noise')
    # in_node.set_shape([None, None, None, channels])

    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()

    # down layers
    for layer in range(0, layers):
        with tf.variable_scope("down_conv_{}".format(str(layer))):
            features = 2 ** layer * features_root

            conv1 = tf.layers.conv2d(in_node, filters=features, kernel_size=filter_size,
                                     padding='same', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=regularizer, name='conv1')
            conv1_bn = tf.layers.batch_normalization(conv1, training=train_phase, name='bn1')
            dropout = tf.layers.dropout(conv1_bn, dropout_rate, training=train_phase, name='dropout')
            conv1_relu = tf.nn.relu(dropout, name='relu1')

            conv2 = tf.layers.conv2d(conv1_relu, filters=features, kernel_size=filter_size,
                                     padding='same', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=regularizer, name='conv2')
            conv2_bn = tf.layers.batch_normalization(conv2, training=train_phase, name='bn2')
            dropout = tf.layers.dropout(conv2_bn, dropout_rate, training=train_phase, name='dropout')
            conv2_relu = tf.nn.relu(dropout, name='relu2')

            dw_h_convs[layer] = conv2_relu

            if layer < layers - 1:
                pools[layer] = tf.layers.max_pooling2d(dw_h_convs[layer], pool_size, strides=pool_size, name='maxpool')
                in_node = pools[layer]

    in_node = dw_h_convs[layers - 1]

    pos = tf.cond(train_phase, lambda: create_position_branch(in_node, batch_size),
                  lambda: np.zeros([batch_size, ], dtype=np.float32), name='position_pred')

    # up layers
    for layer in range(layers - 2, -1, -1):

        with tf.variable_scope("up_conv_{}".format(str(layer))):
            features = 2 ** (layer + 1) * features_root

            h_deconv = tf.layers.conv2d_transpose(in_node, filters=features // 2, kernel_size=pool_size,
                                                  strides=pool_size, use_bias=False,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                  kernel_regularizer=regularizer,
                                                  name='deconv')
            h_deconv_bn = tf.layers.batch_normalization(h_deconv, training=train_phase, name='deconv_bn')
            h_deconv_relu = tf.nn.relu(h_deconv_bn, name='deconv_relu')
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv_relu)
            h_deconv_concat.set_shape([None, None, None, features])
            deconv[layer] = h_deconv_concat

            conv1 = tf.layers.conv2d(h_deconv_concat, filters=features // 2, kernel_size=filter_size,
                                     padding='same', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=regularizer, name='conv1')
            conv1_bn = tf.layers.batch_normalization(conv1, training=train_phase, name='bn1')
            dropout = tf.layers.dropout(conv1_bn, dropout_rate, training=train_phase, name='dropout')
            conv1_relu = tf.nn.relu(dropout, name='relu1')

            conv2 = tf.layers.conv2d(conv1_relu, filters=features // 2, kernel_size=filter_size,
                                     padding='same', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=regularizer, name='conv2')
            conv2_bn = tf.layers.batch_normalization(conv2, training=train_phase, name='bn2')
            dropout = tf.layers.dropout(conv2_bn, dropout_rate, training=train_phase, name='dropout')
            conv2_relu = tf.nn.relu(dropout, name='relu2')

            up_h_convs[layer] = conv2_relu

            in_node = up_h_convs[layer]

    # Output Map
    with tf.variable_scope("output_map"):
        conv = tf.layers.conv2d(in_node, filters=n_class, kernel_size=1, padding='same',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=regularizer,
                                bias_regularizer=regularizer, name='conv')
        up_h_convs["out"] = conv
        output_map = up_h_convs["out"]

    if summaries:
        with tf.variable_scope("summaries"):
            for k in dw_h_convs.keys():
                tf.summary.histogram("dw_convolution_%02d" % k + '/activations', dw_h_convs[k])

            for k in up_h_convs.keys():
                tf.summary.histogram("up_convolution_%s" % k + '/activations', up_h_convs[k])

    return output_map, pos


class UNet(object):
    """
    A U-Net implementation

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost_name: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self, channels=3, n_class=4, batch_size=32, pos_parameter=1e-6,
                 cost_name="cross_entropy", cost_kwargs=None, **net_kwargs):
        if cost_kwargs is None:
            cost_kwargs = {}
        tf.reset_default_graph()

        self.channels = channels
        self.n_class = n_class
        self.batch_size = batch_size
        self.cost_name = cost_name
        self.pos_parameter = pos_parameter
        self.cost_kwargs = cost_kwargs
        self.net_kwargs = net_kwargs
        self.summaries = net_kwargs.get("summaries", True)

        # initialize regularizer
        self.regularizer_type = self.cost_kwargs.pop("regularizer_type", None)
        regularizer = None
        if self.regularizer_type is not None:
            self.regularization_coefficient = self.cost_kwargs.get("regularization_coefficient")

            if self.regularizer_type == 'L2_norm':
                regularizer = tf.contrib.layers.l2_regularizer(scale=self.regularization_coefficient)

            elif self.regularizer_type == 'L1_norm':
                regularizer = tf.contrib.layers.l1_regularizer(scale=self.regularization_coefficient)

        with tf.variable_scope('u_net'):
            self.x = tf.placeholder(tf.float32, shape=[None, None, None, channels], name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, None, None, n_class], name='y')
            self.p = tf.placeholder(tf.float32, shape=[None, ], name='p')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
            self.need_pos = tf.placeholder(tf.bool, name='need_pos')
            self.x = tf.image.random_contrast(self.x, lower=0.5, upper=1)

            logits, pos = create_conv_net(self.x, self.train_phase, dropout_rate=self.dropout_rate,
                                          n_class=self.n_class, regularizer=regularizer,
                                          batch_size=self.batch_size, **net_kwargs)
            self.labels = self.y

        with tf.name_scope('post_processing'):
            self.predictor = self._get_predictor(logits)
            self.segment = self._get_segmentation(self.predictor)

        # get variables and update-ops
        self.trainable_variables = tf.trainable_variables(scope='u_net')
        self.training_variables = tf.global_variables(scope='u_net')
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='u_net')
        # set global step and moving average
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        with tf.name_scope('moving_average'):
            variable_averages = tf.train.ExponentialMovingAverage(decay=0.999, num_updates=self.global_step)
            self.variable_averages_op = variable_averages.apply(self.trainable_variables)
        self.variables_to_restore = variable_averages.variables_to_restore()

        with tf.variable_scope('cost_function'):
            self.cost = tf.cond(self.train_phase,
                                lambda: self._get_cost(logits, self.predictor, self.labels, self.pos_parameter,
                                                       pos=pos, position=self.p,
                                                       need_pos=self.need_pos, regularizer_type=self.regularizer_type),
                                lambda: self._get_cost(logits, self.predictor, self.labels, self.pos_parameter,
                                                       pos=pos, position=self.p,
                                                       need_pos=self.need_pos, regularizer_type=None))
            self.gradients_node = tf.gradients(self.cost, self.trainable_variables, name='gradients')

        with tf.name_scope('metrics'):
            self.correct_pred = tf.equal(tf.argmax(self.predictor, -1), tf.argmax(self.labels, -1))
            self.acc, self.update_acc = tf.metrics.accuracy(tf.argmax(self.labels, -1), tf.argmax(self.predictor, -1),
                                                            name='acc')
            self.sens, self.update_sens = tf.metrics.sensitivity_at_specificity(self.labels[..., 1:],
                                                                                self.predictor[..., 1:], 0.95,
                                                                                num_thresholds=50, name='sens')
            self.spec, self.update_spec = tf.metrics.specificity_at_sensitivity(self.labels[..., 1:],
                                                                                self.predictor[..., 1:], 0.95,
                                                                                num_thresholds=50, name='spec')
            self.auc, self.update_auc = tf.metrics.auc(self.labels[..., 1:], self.predictor[..., 1:], num_thresholds=50,
                                                       name='auc')
            self.dice_score = self._get_dice_score(self.segment, self.labels)

    def _get_predictor(self, logits):
        """
        produce the probability maps from the final feature maps of the network
        """
        return tf.nn.softmax(logits, axis=-1, name='probability_map')

    def _get_segmentation(self, predictor):
        """
        produce the segmentation maps from the probability maps
        """
        return tf.where(tf.equal(tf.reduce_max(predictor, -1, keepdims=True), predictor),
                        tf.ones_like(predictor),
                        tf.zeros_like(predictor), name='segmentation_map')

    def _get_cost(self, logits, probs, labels, pos_parameter, pos, position, need_pos, regularizer_type=None):
        """
        Constructs the cost function based on the cost_name attribution

        :param logits: unscaled log probabilities
        :param probs: probability map produced by softmax layer
        :param labels: one-hot representation of ground-truth
        :param regularizer_type: type of regularization
        Optional arguments are:
        regularization_coefficient: weight of the regularization term
        scale_weight: weights for multi-scale output when computing the combined loss

        :return loss: weighted loss of multi-scale outputs, including the regularization term
        """

        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(labels, [-1, self.n_class])
        flat_probs = tf.reshape(probs, [-1, self.n_class])
        if self.cost_name == 'cross_entropy':
            loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels,
                                                                  name='cross_entropy_map')
            loss = tf.reduce_mean(loss_map)

        elif self.cost_name == 'weighted_cross_entropy':
            loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels,
                                                                  name='cross_entropy_map')
            weight_map = balance_weight_map(flat_labels)
            loss = tf.reduce_mean(loss_map * weight_map)

        elif self.cost_name == "dice_loss":
            dice = 0.
            eps = 1.
            for i in range(self.n_class):
                intersection = tf.reduce_sum(flat_probs[..., i] * flat_labels[..., i])
                union = tf.reduce_sum(flat_probs[..., i] + flat_labels[..., i])
                dice += 1 - (2 * intersection + eps) / (union + eps)
            loss = dice / self.n_class

        elif self.cost_name == "generalized_dice_loss":
            eps = 1.
            class_weight = tf.reduce_sum(flat_labels) / tf.reduce_sum(flat_labels, axis=0)
            intersection = 0.
            union = 0.
            for i in range(self.n_class):
                intersection += tf.reduce_sum(class_weight[i] * tf.reduce_sum(flat_probs[..., i] *
                                                                              flat_labels[..., i]))
                union += tf.reduce_sum(class_weight[i] * tf.reduce_sum(flat_probs[..., i] + flat_labels[..., i]))
            loss = 1 - (2 * intersection + eps) / (union + eps)

        elif self.cost_name == "cross_entropy+dice_loss":
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                      labels=flat_labels),
                                           name='cross_entropy_loss')
            dice_loss = 0.
            eps = 1.
            for i in range(self.n_class):
                intersection = tf.reduce_sum(flat_probs[..., i] * flat_labels[..., i])
                union = tf.reduce_sum(flat_probs[..., i] + flat_labels[..., i])
                dice_loss += 1 - (2 * intersection + eps) / (union + eps)
            loss = cross_entropy + dice_loss / self.n_class

        elif self.cost_name == "weighted_cross_entropy+generalized_dice_loss":
            cross_entropy = tf.reduce_mean(balance_weight_map(flat_labels) *
                                           tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                      labels=flat_labels,
                                                                                      name='cross_entropy_map'))
            eps = 1.
            class_weight = tf.reduce_sum(flat_labels) / tf.reduce_sum(flat_labels, axis=0)
            intersection = 0.
            union = 0.
            for i in range(self.n_class):
                intersection += tf.reduce_sum(class_weight[i] * tf.reduce_sum(flat_probs[..., i] *
                                                                              flat_labels[..., i]))
                union += tf.reduce_sum(class_weight[i] * tf.reduce_sum(flat_probs[..., i] + flat_labels[..., i]))
            dice_loss = 1 - (2 * intersection + eps) / (union + eps)
            loss = cross_entropy + dice_loss / self.n_class

        elif self.cost_name == "exponential_logarithmic":
            eps = 1.
            dice_loss = 0.
            for i in range(self.n_class):
                intersection = tf.reduce_sum(flat_probs[..., i] * flat_labels[..., i])
                union = eps + tf.reduce_sum(flat_probs[..., i] + flat_labels[..., i])
                dice_loss += tf.pow(-tf.log((2 * intersection + eps) / union), .3) / self.n_class
            '''
            cross_weight = tf.reduce_sum(
                flat_labels * tf.pow(tf.reduce_sum(flat_labels) / tf.reduce_sum(flat_labels, axis=0, keepdims=True),
                                     .5), axis=-1)
            cross_loss = tf.reduce_mean(cross_weight * tf.pow(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels), 1.))
            '''
            cross_loss = tf.reduce_mean(tf.pow(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                          labels=flat_labels), 1.))

            loss = 0.8 * dice_loss + 0.2 * cross_loss

        else:
            raise ValueError("Unknown cost function: " % self.cost_name)

        if regularizer_type is not None:
            # add regularization loss
            if regularizer_type in ('L2_norm', 'L1_norm'):
                self.regularization_term = tf.losses.get_regularization_loss(scope='network')

                loss += self.regularization_term
            elif regularizer_type == 'anatomical_constraint_cae':
                with tf.variable_scope('autoencoder', reuse=tf.AUTO_REUSE):
                    segment_codes = create_ae_encoder(self.segment, False, False, batch_size=self.batch_size)
                    segment_decoder = create_ae_decoder(segment_codes, False, n_class=self.n_class)
                    labels_codes = create_ae_encoder(self.labels, False, False, batch_size=self.batch_size)
                    labels_decoder = create_ae_decoder(labels_codes, False, n_class=self.n_class)

                self.ae_variables = tf.global_variables(scope='cost_function/autoencoder')
                self.abs_ae_path = os.path.abspath(self.cost_kwargs.pop('acnn_model_path', './autoencoder_trained'))

                self.regularization_term = tf.nn.l2_loss(segment_decoder - labels_decoder, name='regularization_term')

                loss += self.regularization_coefficient * self.regularization_term

            elif regularizer_type == 'anatomical_constraint_acnn':
                with tf.variable_scope('autoencoder', reuse=tf.AUTO_REUSE):
                    segment_codes = create_ae_encoder(self.segment, False, False, batch_size=self.batch_size)

                    labels_codes = create_ae_encoder(self.labels, False, False, batch_size=self.batch_size)

                self.ae_variables = tf.global_variables(scope='cost_function/autoencoder')
                self.abs_ae_path = os.path.abspath(self.cost_kwargs.pop('acnn_model_path', './autoencoder_trained'))

                self.regularization_term = tf.nn.l2_loss(segment_codes - labels_codes, name='regularization_term')

                loss += self.regularization_coefficient * self.regularization_term

            else:
                raise ValueError("Unknown regularization type: " % self.regularizer_type)

        self.pos_regularization = tf.cond(need_pos, lambda: tf.nn.l2_loss(position - pos, name='position_regularizer'), lambda: 0., name='position_loss')

        loss += pos_parameter * self.pos_regularization

        return loss

    def _get_dice_score(self, segment, label):
        """
        Return the Dice score based only on segmentation results.
        Segmentation is labelled by setting the maximum along classes of predictors to be 1.

        """
        eps = 1.
        dice = 0.
        for i in range(1, self.n_class):
            intersection = tf.reduce_sum(segment[..., i] * label[..., i])
            union = tf.reduce_sum(segment[..., i] + label[..., i])
            dice += 2 * intersection / (union + eps)
        return tf.divide(dice, tf.cast(self.n_class-1, dtype=tf.float32), name='dice_score')

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
        """

        init = tf.global_variables_initializer()
        prediction = []
        with tf.Session(config=config) as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path, var_list=self.variables_to_restore)

            for test_x in x_test:
                y_dummy = np.empty((test_x.shape[0], test_x.shape[1], test_x.shape[2], self.n_class))
                prediction.append(sess.run(self.predictor, feed_dict={self.x: test_x, self.y: y_dummy,
                                                                      self.p: np.empty((self.batch_size, )),
                                                                      self.dropout_rate: 0.0,
                                                                      self.train_phase: False,
                                                                      self.need_pos: False}))

        return prediction

    def save(self, sess, model_path, latest_filename, **kwargs):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        :param latest_filename: Optional name for the protocol buffer file that will contains the list of most recent
        checkpoints.
        """

        saver = tf.train.Saver(**kwargs)
        save_path = saver.save(sess, model_path, latest_filename=latest_filename)
        return save_path

    def restore(self, sess, model_path, **kwargs):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver(**kwargs)
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)


class Trainer(object):
    """
    Trains a U-Net instance.

    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer_name: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer

    """

    def __init__(self, net, batch_size=1, norm_grads=False, optimizer_name="momentum", opt_kwargs=None):
        if opt_kwargs is None:
            opt_kwargs = {}
        self.net = net
        self.batch_size = batch_size
        self.norm_grads = norm_grads
        self.optimizer_name = optimizer_name
        self.opt_kwargs = opt_kwargs
        self.global_step = net.global_step
        self.p_dummy = np.empty((self.batch_size, ))

    def _get_optimizer(self, training_iters, global_step, clip_gradient=False):
        update_ops = self.net.update_ops
        var_list = self.net.trainable_variables
        if self.optimizer_name == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=training_iters,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True, name='learning_rate')

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs)

        elif self.optimizer_name == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate, name='learning_rate')

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                               **self.opt_kwargs)

        else:
            raise ValueError("Unknown optimizer: %s" % self.optimizer_name)

        if clip_gradient:
            gradients, variables = zip(*optimizer.compute_gradients(self.net.cost, var_list=var_list))

            # clip by global norm
            capped_grads, _ = tf.clip_by_global_norm(gradients, 1.0)
            '''
            # clip by individual norm
            capped_grads = [None if grad is None else tf.clip_by_norm(grad, 1.0) for grad in gradients]
            '''
            opt_op = optimizer.apply_gradients(zip(capped_grads, variables), global_step=global_step)
        else:
            opt_op = optimizer.minimize(self.net.cost, global_step=global_step, var_list=var_list)

        with tf.control_dependencies([opt_op]):
            train_op = tf.group([self.net.variable_averages_op, update_ops])

        return optimizer, train_op

    def _initialize(self, training_iters, clip_gradient, model_path, restore, prediction_path):
        global_step = self.net.global_step

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]),
                                               name='norm_gradients')

        if self.net.summaries and self.norm_grads:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        # create summary protocol buffers for training metrics
        with tf.name_scope('train_metrics_summary'):
            tf.summary.scalar('Training_Loss', self.net.cost)
            if self.net.regularizer_type is not None:
                tf.summary.scalar('Training_Regularization_Loss', self.net.regularization_term)
            tf.summary.scalar('Training_Accuracy', self.net.acc)
            tf.summary.scalar('Training_AUC', self.net.auc)
            tf.summary.scalar('Training_Sensitivity', self.net.sens)
            tf.summary.scalar('Training_Specificity', self.net.spec)
            tf.summary.scalar('Training_Dice_score', self.net.dice_score)

        # initialize optimizer
        with tf.name_scope('optimizer'):
            self.optimizer, self.train_op = self._get_optimizer(training_iters, global_step, clip_gradient)

        # create a summary protocol buffer for learning rate
        with tf.name_scope('lr_summary'):
            tf.summary.scalar('learning_rate', self.learning_rate_node)

        # Merges summaries in the default graph
        self.summary_op = tf.summary.merge_all()

        # create an op that initializes all training variables
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        model_path = os.path.abspath(model_path)

        # remove the previous directory for model storing and validation prediction
        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(model_path))
            shutil.rmtree(model_path, ignore_errors=True)

        # create a new directory for model storing and validation prediction
        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(model_path):
            logging.info("Allocating '{:}'".format(model_path))
            os.makedirs(model_path)

        return init

    def train(self, train_data_provider, val_data_provider, train_original_data_provider, validation_batch_size, model_path, training_iters=10,
              epochs=100, dropout=0.75, clip_gradient=False, display_step=1, restore=True, write_graph=False,
              prediction_path='validation_prediction'):
        """
        Launches the training process

        :param train_data_provider: callable returning training data
        :param val_data_provider: callable returning validation data
        :param validation_batch_size: number of data for validation
        :param model_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param clip_gradient: whether to apply gradient clipping
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """

        save_path = os.path.join(model_path, "best_model.ckpt")
        goon_path = os.path.join(model_path, "goon_model.ckpt")

        init = self._initialize(training_iters, clip_gradient, model_path, restore, prediction_path)

        with tf.Session(config=config) as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, model_path, "graph.pb", False)

            # initialization
            sess.run(init)

            # ACNN regularization
            if self.net.regularizer_type == 'anatomical_constraint':
                ae_ckpt = tf.train.get_checkpoint_state(self.net.abs_ae_path)
                if ae_ckpt and ae_ckpt.model_checkpoint_path:
                    logging.info("Model restored from file: {:}".format(ae_ckpt.model_checkpoint_path))
                    # print([v.name for v in self.net.ae_variables])
                    ae_var_list = dict((v.name.lstrip('cost_function/').rstrip(':0'), v) for v in self.net.ae_variables)
                    self.net.restore(sess, ae_ckpt.model_checkpoint_path, var_list=ae_var_list)

            # restore model
            if restore:
                ckpt = tf.train.get_checkpoint_state(model_path, latest_filename='goon_checkpoint')
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path,
                                     var_list=self.net.training_variables + [tf.train.get_global_step()])

            # create summary writer for training summaries
            summary_writer = tf.summary.FileWriter(model_path, graph=sess.graph)

            # read validation data
            test_x, test_y, test_affine, _ = val_data_provider(validation_batch_size)
            # read the original train data
            train_x, train_y, train_affine, _ = train_original_data_provider(25)
            # visualize performance on validation data
            self.store_prediction(sess, test_x, test_y, test_affine)

            test_acc = np.array([])
            test_dice = np.array([])
            test_auc = np.array([])
            test_sens = np.array([])
            test_spec = np.array([])

            if epochs == 0:
                return save_path, test_acc, test_dice, test_auc, test_sens, test_spec

            logging.info("Start U-net optimization based on loss function: {} and regularizer type: {}".format(
                self.net.cost_name, self.net.regularizer_type))
            if self.net.regularizer_type is not None:
                logging.info("Current regularization coefficient: {}".format(self.net.regularization_coefficient))

            lr = 0.
            avg_gradients = None
            for epoch in range(epochs):
                total_loss = 0.
                for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                    # read training data
                    batch_x, batch_y, _, batch_position = train_data_provider(self.batch_size)

                    # get output shape
                    prediction = sess.run(self.net.predictor, feed_dict={self.net.x: batch_x,
                                                                         self.net.y: batch_y,
                                                                         self.net.p: batch_position,
                                                                         self.net.dropout_rate: 0.,
                                                                         self.net.train_phase: False,
                                                                         self.net.need_pos: False})
                    pred_shape = prediction.shape

                    # optimization operation (back-propagation)

                    _, loss, lr, gradients = sess.run([self.train_op, self.net.cost, self.learning_rate_node,
                                                       self.net.gradients_node],
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                                                 self.net.p: batch_position,
                                                                 self.net.dropout_rate: dropout,
                                                                 self.net.train_phase: True,
                                                                 self.net.need_pos: True})

                    # add normalized gradients to summaries
                    if self.net.summaries and self.norm_grads:
                        avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
                        norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                        self.norm_gradients_node.assign(norm_gradients).eval()

                    # display mini-batch statistics
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x,
                                                    util.crop_to_shape(batch_y, pred_shape))

                    total_loss += loss

                # display epoch statistics
                self.output_epoch_stats(epoch, total_loss, training_iters, lr)

                # save the current model
                model_path_per_epoch = os.path.join(model_path, "model_{}.ckpt".format(epoch))
                self.net.save(sess, model_path_per_epoch, latest_filename='model_{}_checkpoint'.format(epoch))
                self.net.save(sess, goon_path, latest_filename='goon_checkpoint')

                # visualize and display validation performance and metrics
                acc, dice, auc, sens, spec = self.store_prediction(sess, test_x, test_y, test_affine)
                print('#################### result of original train data ######################')
                self.store_prediction(sess, train_x, train_y, train_affine)

                # save the current model if it is the best one hitherto
                if epoch > 0 and dice > np.max(test_dice):
                    save_path = self.net.save(sess, save_path, latest_filename='best_checkpoint')

                # store the validation metrics
                test_acc = np.hstack((test_acc, acc))
                test_dice = np.hstack((test_dice, dice))
                test_auc = np.hstack((test_auc, auc))
                test_sens = np.hstack((test_sens, sens))
                test_spec = np.hstack((test_spec, spec))
            logging.info("Optimization Finished!")

            return save_path, test_acc, test_dice, test_auc, test_sens, test_spec

    def store_prediction(self, sess, batch_x, batch_y, batch_affine):

        n = len(batch_y)
        loss = np.zeros([n])
        dice = np.zeros([n])
        batch_pred = []

        sess.run(tf.local_variables_initializer())
        for i in range(n):

            pred = sess.run(self.net.predictor, feed_dict={self.net.x: batch_x[i],
                                                           self.net.y: batch_y[i],
                                                           self.net.p: self.p_dummy,
                                                           self.net.dropout_rate: 0.,
                                                           self.net.train_phase: False,
                                                           self.net.need_pos: False})
            pred_shape = pred.shape
            batch_pred.append(pred)

            loss[i], dice[i] = sess.run([self.net.cost, self.net.dice_score],
                                        feed_dict={self.net.x: batch_x[i],
                                                   self.net.y: util.crop_to_shape(batch_y[i], pred_shape),
                                                   self.net.p: self.p_dummy,
                                                   self.net.dropout_rate: 0.,
                                                   self.net.train_phase: False,
                                                   self.net.need_pos: False})

            batch_x[i] = np.expand_dims(batch_x[i], axis=0).transpose((0, 2, 3, 1, 4))
            batch_y[i] = np.expand_dims(batch_y[i], axis=0).transpose((0, 2, 3, 1, 4))
            batch_pred[i] = np.expand_dims(batch_pred[i], axis=0).transpose((0, 2, 3, 1, 4))

        acc, auc, sens, spec = sess.run([self.net.acc, self.net.auc, self.net.sens, self.net.spec])
        logging.info(
            "Validation Error= {:.2f}%, Loss= {:.4f}, Dice score= {:.4f}, AUC= {:.4f}, Sensitivity= {:.2f}%, "
            "Specificity= {:.2f}% ".format(
                (1 - acc) * 100, np.mean(loss), np.mean(dice), auc, sens * 100, spec * 100))
        util.save_prediction(batch_x, batch_y, batch_pred, self.prediction_path)
        util.save_prediction_1(batch_pred, batch_affine, self.prediction_path)

        for i in range(n):
            batch_x[i] = np.squeeze(batch_x[i], axis=0).transpose((2, 0, 1, 3))
            batch_y[i] = np.squeeze(batch_y[i], axis=0).transpose((2, 0, 1, 3))

        return acc, np.mean(dice), auc, sens, spec

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info(
            "Epoch {:}, Average Loss: {:.4f}, learning rate: {:.1e}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        sess.run(tf.local_variables_initializer())
        summary_str, loss, dice = sess.run([self.summary_op, self.net.cost,
                                            self.net.dice_score],
                                            feed_dict={self.net.x: batch_x, self.net.y: batch_y,
                                                       self.net.p: self.p_dummy,
                                                       self.net.dropout_rate: 0.,
                                                       self.net.train_phase: True,
                                                       self.net.need_pos: False})
        acc, auc, sens, spec = sess.run([self.net.acc, self.net.auc, self.net.sens, self.net.spec])
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info(
            "Iter {:}, Mini-batch Loss= {:.4f}, Accuracy= {:.2f}%, Dice score= {:.4f}, "
            "AUC= {:.4f}, Sensitivity= {:.2f}%, Specificity= {:.2f}%".format(
                step, loss, acc * 100, dice, auc, sens * 100, spec * 100))


def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (gradients[i] / (step + 1))

    return avg_gradients


def acc_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and labels.
    :param predictions: list of output predictions
    :param labels: list of ground truths
    """
    assert len(predictions) == len(labels), "Number of predictions and labels don't equal."
    err = np.array([])
    n = len(predictions)
    for i in range(n):
        err = np.hstack((err, (100.0 * np.average(
            np.argmax(predictions[i], -1) == np.argmax(util.crop_to_shape(labels[i], predictions[i].shape), -1)))))
    return err


def auc_score(predictions, labels):
    """
    Return the auc score based on dense predictions and labels.
    :param predictions: list of output predictions
    :param labels: list of ground truths
    """
    assert len(predictions) == len(labels), "Number of predictions and labels don't equal."
    auc = np.array([])
    n = len(predictions)
    n_class = labels[0].shape[-1]
    for i in range(n):
        flat_score = np.reshape(predictions[i], [-1, n_class])
        flat_true = np.reshape(util.crop_to_shape(labels[i], predictions[i].shape), [-1, n_class])
        auc = np.hstack((auc, roc_auc_score(flat_true, flat_score)))
    return auc


# def dice_score(predictions, labels):
#     """
#     Return the dice score based on dense predictions and labels.
#     :param predictions: list of output predictions
#     :param labels: list of ground truths
#     """
#     assert len(predictions) == len(labels), "Number of predictions and labels don't equal."
#     dice = np.array([])
#     n = len(predictions)
#     eps = 1.
#     for i in range(n):
#         pred = np.array(predictions[i])
#         label = np.array(labels[i])
#         mask = np.where(np.equal(np.max(pred, -1, keepdims=True), pred),
#                         np.ones_like(pred),
#                         np.zeros_like(pred))
#         intersection = np.sum(mask[..., 1:] * util.crop_to_shape(label, pred.shape)[..., 1:])
#         union = eps + np.sum(mask[..., 1:] + util.crop_to_shape(label, pred.shape)[..., 1:])
#         dice = np.hstack((dice, 2 * intersection / union))
#     return dice


def dice_score(predictions, labels):
    """
    Return the dice score based on dense predictions and labels.
    :param predictions: list of output predictions
    :param labels: list of ground truths
    """
    assert len(predictions) == len(labels), "Number of predictions and labels don't equal."
    n_class = labels[0].shape[-1]
    dice = np.array([])
    n = len(predictions)
    eps = 1.
    for i in range(n):
        pred = np.array(predictions[i])
        label = util.crop_to_shape(np.array(labels[i]), pred.shape)
        mask = np.where(np.equal(np.max(pred, -1, keepdims=True), pred),
                        np.ones_like(pred),
                        np.zeros_like(pred))
        d = 0.
        for k in range(1, n_class):
            numerator = 2 * np.sum(mask[..., k] * label[..., k])
            denominator = np.sum(mask[..., k] + label[..., k])
            d += numerator / (eps + denominator)

        dice = np.hstack((dice, d / (n_class-1)))
    return dice


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V


def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)]
