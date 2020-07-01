# -*- coding: utf-8 -*-
"""
Created on Tue Nov 6 10:08:21 2018

@author: Xinzhe Luo
"""

import tensorflow as tf
import numpy as np
import logging
import os
import shutil
from core.util import crop_to_shape

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def create_ae_encoder(x, training, summaries=True, batch_size=32):
    """
    Introduce the encoder part of the auto-encoder for anatomical constraint.
    
    :param x: Input label tensor, expected dim [batch_size, height, width, n_class],
              here roi = [240, 240]
    :param training: Whether to return the output in training mode or in inference mode.
    :param summaries: Flag if summaries should be created.
    
    :returns: codes (compact representation of labels); the tensor shape of the last encoder convolution
    """

    # dw_layers
    with tf.variable_scope('encoder'):
        in_node = x
        # [batch_size, 240, 240, n_class]
        with tf.variable_scope('dw_conv_layer1'):
            conv1 = tf.layers.conv2d(in_node, filters=16, kernel_size=3, strides=2, padding='same',
                                     use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv1')
            bn1 = tf.layers.batch_normalization(conv1, training=training, name='bn1')
            relu1 = tf.nn.relu(bn1, name='relu1')

            conv2 = tf.layers.conv2d(relu1, filters=16, kernel_size=3, use_bias=False, padding='same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv2')
            bn2 = tf.layers.batch_normalization(conv2, training=training, name='bn2')
            relu2 = tf.nn.relu(bn2, name='relu2')

            in_node = relu2
            # [batch_size, 120, 120, 16]

        with tf.variable_scope('dw_conv_layer2'):
            conv1 = tf.layers.conv2d(in_node, filters=32, kernel_size=3, strides=2, padding='same',
                                     use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv1')
            bn1 = tf.layers.batch_normalization(conv1, training=training, name='bn1')
            relu1 = tf.nn.relu(bn1, name='relu1')

            conv2 = tf.layers.conv2d(relu1, filters=32, kernel_size=3, use_bias=False, padding='same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv2')
            bn2 = tf.layers.batch_normalization(conv2, training=training, name='bn2')
            relu2 = tf.nn.relu(bn2, name='relu2')

            in_node = relu2
            # [batch_size, 60, 60, 32]

        with tf.variable_scope('dw_conv_layer3'):
            conv1 = tf.layers.conv2d(in_node, filters=64, kernel_size=3, strides=2, padding='same',
                                     use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv1')
            bn1 = tf.layers.batch_normalization(conv1, training=training, name='bn1')
            relu1 = tf.nn.relu(bn1, name='relu1')

            conv2 = tf.layers.conv2d(relu1, filters=64, kernel_size=3, use_bias=False, padding='same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv2')
            bn2 = tf.layers.batch_normalization(conv2, training=training, name='bn2')
            relu2 = tf.nn.relu(bn2, name='relu2')

            in_node = relu2
            # [batch_size, 30, 30, 64]

        with tf.variable_scope('dw_conv_layer4'):
            conv = tf.layers.conv2d(in_node, filters=1, kernel_size=3, strides=(3, 3), padding='same',
                                    use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='conv')
            bn = tf.layers.batch_normalization(conv, training=training, name='bn')
            relu = tf.nn.relu(bn, name='relu')

            in_node = relu
            # [batch_size, 10, 10, 1]

        with tf.variable_scope('fc_layer'):
            in_node = tf.reshape(in_node, [batch_size, 100])
            codes = tf.layers.dense(in_node, units=64,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='codes')
            # [batch_size, 64]
    
    if summaries:    
        with tf.name_scope('summaries'):
            tf.summary.histogram('codes', codes)

    return codes


def create_ae_decoder(x, training, n_class):
    """
    Construct the decoder part of the auto-encoder for anatomical constraint.

    :param x: Input codes tensor, expected a vector of length of codes. [batch_size, 64]
    :param training: Whether to return the output in training mode or in inference mode.
    :param n_class: The number of output classes.
    :return: Decodes representing the reconstructed label maps.
    """
    with tf.variable_scope('decoder'):
        in_node = x
        with tf.variable_scope('fc_layer'):
            fc = tf.layers.dense(in_node, units=100, activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='fc')

            in_node = tf.reshape(fc, [-1, 10, 10, 1])
            # [batch_size, 10, 10, 1]

        with tf.variable_scope('up_conv_layer4'):
            deconv = tf.layers.conv2d_transpose(in_node, filters=64, kernel_size=7,
                                                strides=(3, 3), padding='same', use_bias=False,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name='deconv')
            bn1 = tf.layers.batch_normalization(deconv, training=training, name='bn1')
            relu1 = tf.nn.relu(bn1, name='relu1')

            conv = tf.layers.conv2d(relu1, filters=64, kernel_size=3, padding='same',
                                    use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='conv')
            bn2 = tf.layers.batch_normalization(conv, training=training, name='bn2')
            relu2 = tf.nn.relu(bn2, name='relu2')

            in_node = relu2
            # [batch_size, 30, 30, 64]

        with tf.variable_scope('up_conv_layer3'):
            deconv = tf.layers.conv2d_transpose(in_node, filters=32, kernel_size=4,
                                                strides=2, padding='same', use_bias=False,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name='deconv')
            bn1 = tf.layers.batch_normalization(deconv, training=training, name='bn1')
            relu1 = tf.nn.relu(bn1, name='relu1')

            conv = tf.layers.conv2d(relu1, filters=32, kernel_size=3, padding='same',
                                    use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='conv')
            bn2 = tf.layers.batch_normalization(conv, training=training, name='bn2')
            relu2 = tf.nn.relu(bn2, name='relu2')

            in_node = relu2
            # [batch_size, 60, 60, 32]

        with tf.variable_scope('up_conv_layer2'):
            deconv = tf.layers.conv2d_transpose(in_node, filters=16, kernel_size=4,
                                                strides=2, padding='same', use_bias=False,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name='deconv')
            bn1 = tf.layers.batch_normalization(deconv, training=training, name='bn1')
            relu1 = tf.nn.relu(bn1, name='relu1')

            conv = tf.layers.conv2d(relu1, filters=16, kernel_size=3, padding='same',
                                    use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='conv')
            bn2 = tf.layers.batch_normalization(conv, training=training, name='bn2')
            relu2 = tf.nn.relu(bn2, name='relu2')

            in_node = relu2
            # [batch_size, 120, 120, 16]

        with tf.variable_scope('up_conv_layer1'):
            deconv = tf.layers.conv2d_transpose(in_node, filters=16, kernel_size=4,
                                                strides=2, padding='same', use_bias=False,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name='deconv')
            bn1 = tf.layers.batch_normalization(deconv, training=training, name='bn1')
            relu1 = tf.nn.relu(bn1, name='relu1')

            conv = tf.layers.conv2d(relu1, filters=n_class, kernel_size=3, padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='conv')
            # [batch_size, 240, 240, 4]

    return conv


class AutoEncoder(object):
    """
    An anatomical constraint auto-encoder implementation.
    
    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param batch_size: size of training batch
    """
    
    def __init__(self, channels=1, n_class=4, batch_size=1, optimizer="adam", cost_kwargs=None, opt_kwargs=None):
        tf.reset_default_graph()

        if cost_kwargs is None:
            cost_kwargs = {}
        if opt_kwargs is None:
            opt_kwargs = {}
        
        self.channels = channels
        self.n_class = n_class
        self.batch_size = batch_size
        self.cost_kwargs = cost_kwargs
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs

        self.__labels = tf.placeholder("float", shape=[self.batch_size, 240, 240, n_class], name='labels')
        self.__train_phase = tf.placeholder(tf.bool, name='train_phase')
        
        with tf.variable_scope('autoencoder'):
            self.__codes = create_ae_encoder(self.__labels, self.__train_phase, batch_size=self.batch_size)
            # with shape [batch_size, 64]
            self.__decodes = create_ae_decoder(self.__codes, self.__train_phase, self.n_class)
            self.predictor = self._get_predictor(self.__decodes)
            
        with tf.name_scope('cost_function'):
            self.__cost = self._get_cost(self.__decodes, self.__labels)

    def _get_predictor(self, logits):
        """
        produce the probability maps from the final feature maps of the network
        """
        return tf.nn.softmax(logits, axis=-1, name='probability_map')

    def _get_cost(self, logits, labels):
        """
        Construct the loss function of the auto-encoder.
        """
        
        loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels,
                                                              name='cross_entropy_map')
        cross_entropy = tf.reduce_mean(loss_map, name='cross_entropy')
        
        return cross_entropy
    
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

    def _get_optimizer(self, training_iters, global_step):
        """
        Construct optimizer based on the type of optimization
        
        :param training_iters: decay step used for learning rate in momentum optimization
        :param global_step: total frequency of optimization operation
        
        :return train_op: optimization operation        
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        var_list = tf.trainable_variables(scope='autoencoder')
        
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)

            self.__learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                   global_step=global_step,
                                                                   decay_steps=training_iters,
                                                                   decay_rate=decay_rate,
                                                                   staircase=True, name='learning_rate')
                
            with tf.control_dependencies(update_ops):
                train_op = tf.train.MomentumOptimizer(learning_rate=self.__learning_rate_node, momentum=momentum,
                                                      **self.opt_kwargs).minimize(self.__cost,
                                                                                  global_step=global_step,
                                                                                  var_list=var_list)
                
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.__learning_rate_node = tf.Variable(learning_rate, name="learning_rate")
            
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=self.__learning_rate_node,
                                                  **self.opt_kwargs).minimize(self.__cost,
                                                                              global_step=global_step,
                                                                              var_list=var_list)

        else:
            raise ValueError("Unknown optimizer type: " % self.optimizer)

        return train_op

    def _initialize(self, training_iters, model_path, restore):
        """
        initialize optimization operation and model variables; 
        create model saving direction and summary operation
        
        :param training_iters: decay step used for learning rate in momentum optimization
        :param model_path: path to file system location
        :param restore: whether to restore a previously trained model
        
        :return init: global variables initializer
        """
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
        tf.summary.scalar('loss', self.__cost)
        self.summary_op = tf.summary.merge_all()
        
        self.__optimizer = self._get_optimizer(training_iters, global_step)
        
        init = tf.global_variables_initializer()
        
        abs_model_path = os.path.abspath(model_path)        
        if not restore:
            print("Removing '{:}'".format(abs_model_path))
            shutil.rmtree(abs_model_path, ignore_errors=True)
        
        if not os.path.exists(model_path):
            print("Allocating '{:}'".format(abs_model_path))
            os.makedirs(abs_model_path)
            
        return init

    def train(self, data_provider, model_path, training_iters=10, epochs=10, display_step=1, restore=False):
        """
        Lauches the training process

        :param data_provider: callable returning training data
        :param model_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        """
        
        save_path = os.path.join(model_path, 'model.ckpt')
        if epochs == 0:
            return save_path
        
        init = self._initialize(training_iters, model_path, restore)

        with tf.Session() as sess:
            sess.run(init)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    var_list = tf.global_variables(scope='autoencoder') + self.__optimizer.variables() + [
                        tf.train.get_global_step()]
                    self.restore(sess, ckpt.model_checkpoint_path, var_list=var_list)

            summary_writer = tf.summary.FileWriter(model_path, graph=sess.graph)

            logging.info("Start Optimization!")

            for epoch in range(epochs):
                total_loss = 0.
                for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                    _, batch_y, _, _ = data_provider(self.batch_size)
                    
                    decodes = sess.run(self.__decodes, feed_dict={self.__labels: batch_y,
                                                                  self.__train_phase: False})
    
                    _, batch_loss = sess.run((self.__optimizer, self.__cost),
                                             feed_dict={self.__labels: crop_to_shape(batch_y, decodes.shape),
                                                        self.__train_phase: True})
    
                    if step % display_step == 0:
                        logging.info("Iteration {:}, Mini-batch loss= {:.4f}".format(step, batch_loss))
                        summary_str = sess.run(self.summary_op, feed_dict={self.__labels: batch_y,
                                                                           self.__train_phase: True})
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()
                    
                    total_loss += batch_loss

                logging.info("Epoch {:}, Average mini-batch loss= {:.4f}".format(epoch, total_loss / training_iters))
                
                save_path = self.save(sess, save_path, "checkpoint")
                
            logging.info("Optimization Finished!")
        
        return save_path
