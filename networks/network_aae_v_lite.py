#
# ---------------------------------------------------------
# Copyright 2018-present (c) Automatos Studios. All Rights Reserved.
# ---------------------------------------------------------
# 
# @file     network_aae_v_lite.py
# @author   cosmoplankton@automatos.studio
#

"""
THIS IS THE LITE IMPLEMENTATION FOR DEBUGGING.
"""


import numpy as np
import tensorflow as tf

from networks import network_aae_base as base
from utils import global_logging as logging
LOGGER = logging.get_logger()

conv_kernel_size = [4, 4, 4] # 3D-kernel as our input is a 3D-image
conv_stride = 2 # make this available as an user-setting


class Encoder(base.Network):
  """
  debug implementation
  """

  def __init__(self, model, params):
    """
    Args:
      model: parent model object.
      params: dict() of parameters.
    """
    super().__init__(model, params)
    self._var_scope = "encoder"
    self._lambda_end_d = 1.0
    self._n_classes = 0
    self._n_latent_dim = 16

    if "n_classes" in params.keys():
      self._n_classes = params["n_classes"]

    if "lambda_d" in params.keys():
      self._lambda_enc_d = params["lambda_d"]

    if "latent_dim" in params.keys():
      self._n_latent_dim = params["latent_dim"]

    self._init_optimizer()


  def _network(self, input):
    """
    forward network.
    """

    # convolution layer
    def layer (input, n_filters):
      convolution = tf.layers.conv3d(
          inputs= input,
          filters= n_filters,
          kernel_size= conv_kernel_size,
          strides= conv_stride,
          activation= tf.nn.relu,
          padding= 'SAME')
      b_norm = tf.layers.batch_normalization(inputs= convolution)

      return b_norm

    # [TODO] make this a input to the network.
    sequential = [1, 2, 4, 8, 16, 16]
    output= input
    for n in sequential:
      output = layer(output, n)
      LOGGER.debug('[LAYER] [N_FILTER: {0}] {1}'.format(n, output))

    outer_dim = np.prod(output.shape[1:])
    flatten = tf.reshape(output, [-1, outer_dim])

    network_output = tf.layers.dense(inputs= flatten, units= self._n_latent_dim)

    return network_output


  def _loss(self, _data):
    """
    prepare the loss operation
    """
    # use common ones and zeros.
    ones = np.ones(shape=[self._batch_size_shard, self._n_classes + 1])
    y_z_real = tf.constant(ones)

    minimax_error = tf.losses.sigmoid_cross_entropy(
        multi_class_labels= y_z_real,
        logits= _data['y_Hat_z_fake'],
        weights= 1.0)

    loss_op = _data['x_recons_error'] + \
        tf.reduce_mean(tf.scalar_mul(
            scalar= self._lambda_enc_d,
            x= minimax_error))

    return loss_op



class Decoder(base.Network):
  """
  debug implementation
  """

  def __init__(self, model, params):
    """
    Args:
      model: parent model object.
      params: dict() of parameters.
    """
    super().__init__(model, params)
    self._var_scope = "decoder"
    self._lambda_dec_d = 1.0
    self._n_classes = 0
    self._n_channels = 1

    if "n_classes" in params.keys():
      self._n_classes = params["n_classes"]

    if "lambda_d" in params.keys():
      self._lambda_dec_d = params["lambda_d"]

    self._init_optimizer()


  def _network(self, input):
    """
    forward netwrok.
    """
    # reshape before connecting to convolution layer
    output = tf.layers.dense(inputs=input, units=16*1*1*1)
    output = tf.reshape(output, shape=[-1, 1, 1, 1, 16])

    # convolution transpose layer
    def layer (input, n_filters):
      convolution = tf.layers.conv3d_transpose(
          inputs= input,
          filters= n_filters,
          kernel_size= conv_kernel_size,
          strides= conv_stride,
          activation= tf.nn.relu,
          padding= 'SAME')
      b_norm = tf.layers.batch_normalization(inputs= convolution)

      return b_norm

    # [TODO] make this a input to the network.
    sequential = [16, 8, 4, 2, 1]
    for n in sequential:
      output = layer(output, n)
      LOGGER.debug('[LAYER] [N_FILTER: {0}] {1}'.format(n, output))

    network_output = tf.layers.conv3d_transpose(
        inputs= output,
        filters= self._n_channels,
        kernel_size= conv_kernel_size,
        strides= conv_stride,
        activation= tf.nn.sigmoid,
        padding= 'SAME')

    return network_output


  def _loss(self, _data):
    """
    prepare the loss operation
    """
    ones = np.ones(shape=[self._batch_size_shard, self._n_classes + 1])
    y_x_real = tf.constant(ones)

    minimax_error_fake = tf.losses.sigmoid_cross_entropy(
        multi_class_labels= y_x_real,
        logits=_data['y_Hat_x_fake'],
        weights=0.25)

    minimax_error_decode = tf.losses.sigmoid_cross_entropy(
        multi_class_labels= y_x_real,
        logits= _data['y_Hat_x_decode'],
        weights= 0.25)

    loss_op = _data['x_recons_error'] + \
        tf.reduce_mean(tf.scalar_mul(
            scalar= self._lambda_dec_d,
            x= minimax_error_fake)) + \
        tf.reduce_mean(tf.scalar_mul(
            scalar= self._lambda_dec_d,
            x= minimax_error_decode))

    return loss_op


class EncoderD(base.Network):
  """
  debug implementation
  """

  def __init__(self, model, params):
    """
    Args:
      model: parent model object.
      params: dict() of parameters.
    """
    super().__init__(model, params)
    self._var_scope = "encoder_d"
    self._n_classes = 0

    if "n_classes" in params.keys():
      self._n_classes = params["n_classes"]

    self._init_optimizer()


  def _network(self, input):
    """
    forward network.
    """

    # dense layer
    def layer (input, units, add_b_norm= True):
      output = tf.layers.dense(
          inputs= input,
          units= units,
          activation= tf.nn.leaky_relu)

      if add_b_norm:
        output= tf.layers.batch_normalization(inputs= output)

      return output

    # [TODO] make this a input to the network.
    sequential = [(16, False), (16, True), (8, True)]
    output = input
    for n in sequential:
      output = layer(output, n[0], n[1])
      LOGGER.debug('[LAYER] [UNITS: {0}] {1}'.format(n, output))

    network_output = tf.layers.dense(inputs= output, units= self._n_classes + 1, activation= tf.nn.sigmoid)

    return network_output


  def _loss(self, _data):
    """
    prepare the loss operation
    """
    ones = np.ones(shape=[self._batch_size_shard, self._n_classes + 1])
    zeros = np.zeros(shape=[self._batch_size_shard, self._n_classes + 1])

    y_z_real = tf.constant(ones)
    y_z_fake = tf.constant(zeros)

    y_z_fake_error = tf.losses.sigmoid_cross_entropy(
        multi_class_labels= y_z_fake,
        logits= _data['y_Hat_z_fake'],
        weights= 0.5)

    y_z_real_error = tf.losses.sigmoid_cross_entropy(
        multi_class_labels= y_z_real,
        logits= _data['y_Hat_z_real'],
        weights= 0.5)

    loss_op = tf.reduce_mean(y_z_fake_error + y_z_real_error)

    return loss_op


class DecoderD(base.Network):
  """
  debug implementation
  """

  def __init__(self, model, params):
    """
    Args:
      model: parent model object.
      params: dict() of parameters.
    """
    super().__init__(model, params)
    self._var_scope = "decoder_d"
    self._n_classes = 0

    if "n_classes" in params.keys():
      self._n_classes = params["n_classes"]

    self._init_optimizer()


  def _network(self, input):
    """
    forward network.
    """
    # add noise to the input, noise_std_sigma = 0.01

    # convolution layer
    def layer (input, n_filters):
      convolution = tf.layers.conv3d(
          inputs= input,
          filters= n_filters,
          kernel_size= conv_kernel_size,
          strides= conv_stride,
          activation= tf.nn.leaky_relu,
          padding= 'SAME')
      b_norm = tf.layers.batch_normalization(inputs= convolution)

      return b_norm

    # [TODO] make this a input to the network.
    sequential = [1, 2, 4, 8, 16, 16]
    output= input
    for n in sequential:
      output = layer(output, n)
      LOGGER.debug('[LAYER] [N_FILTER: {0}] {1}'.format(n, output))

    outer_dim = np.prod(output.shape[1:])
    flatten = tf.reshape(output, [-1, outer_dim])

    network_output = tf.layers.dense(inputs= flatten, units= self._n_classes + 1)

    return network_output


  def _loss(self, _data):
    """
    prepare the loss operation
    """
    zeros = np.zeros(shape=[self._batch_size_shard, self._n_classes + 1])
    ones = np.ones(shape=[self._batch_size_shard, self._n_classes + 1])

    y_x_real = tf.constant(ones)
    y_x_fake = tf.constant(zeros)
    y_x_decode = tf.constant(zeros)

    y_x_fake_error = tf.losses.sigmoid_cross_entropy(
        multi_class_labels= y_x_real,
        logits= _data['y_Hat_x_fake'],
        weights= 0.25)

    y_x_real_error = tf.losses.sigmoid_cross_entropy(
        multi_class_labels= y_x_fake,
        logits= _data['y_Hat_x_real'],
        weights= 0.5)

    y_x_decode_error = tf.losses.sigmoid_cross_entropy(
        multi_class_labels= y_x_decode,
        logits= _data['y_Hat_x_decode'],
        weights= 0.25)

    loss_op = tf.reduce_mean(y_x_fake_error + y_x_real_error + y_x_decode_error)

    return loss_op

