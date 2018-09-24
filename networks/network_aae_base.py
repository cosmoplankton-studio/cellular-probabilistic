#
# ---------------------------------------------------------
# Copyright 2018-present (c) Automatos Studios. All Rights Reserved.
# ---------------------------------------------------------
# 
# @file     network_aae_base.py
# @author   cosmoplankton@automatos.studio
#

"""
Abstract base implementation for the component networks.
e.g. encoder, decoder, encoder-discrminator and decoder-discriminator.
"""


import abc

import tensorflow as tf

from utils import global_logging as logging
LOGGER = logging.get_logger()


#----------------------------------------------------------
# 'Convolution' output volume calculation (along one dimension, assuming uniform dimensions)
# [source] - http://cs231n.github.io/convolutional-networks/
#----------------------------------------------------------
# O[dim-0] = (W - F + 2P)/S + 1
# W = input[dim-0], F = kernel-size[dim-0], P = zero-pad[dim-0], S = stride[dim-0], K = n-kernel
# >>> input =  W x W x n-channels
# >>> output = O x O x n-kernel
#----------------------------------------------------------


class Network(abc.ABC):
  """
  Base class for the component networks.
  """
  def __init__(self, model, params):
    """
    Args:
      model: parent model object.
      params: dict() of parameters with below allowed KEYS.
          "input_format" - channel firts/ channel last input format.
          "lrate" - leraning rate of the network.
          "reuse" - reuse variables.
          "batch_size" - batch size for training (this will get distributed if multiple GPUs).
          "n_gpu" - number of GPUs.
          "sync_replicas" - sync replica graphs for synchronous distributed training.
          "n_replicas" - number of replica workers.
          "is_chief" - is this worker the chief.
    """
    self._model = model
    self._global_step = None
    self._input_shape = []
    self._var_scope = "networkbase"

    self._input_format = 'channels_last'
    self._learning_rate = 0.001
    self._reuse = False
    self._batch_size = 1000
    self._n_gpu = 1
    self._sync_replicas = False
    self._n_replicas = 1
    self._is_chief = True

    if "input_format" in params.keys():
      self._n_classes = params["input_format"]

    if "lrate" in params.keys():
      self._learning_rate = params["lrate"]

    if "reuse" in params.keys():
      self._reuse = params["reuse"]

    if "batch_size" in params.keys():
      self._batch_size = params["batch_size"]

    if "n_gpu" in params.keys():
      self._n_gpu = params["n_gpu"]

    if "sync_replicas" in params.keys():
      self._sync_replicas = params["sync_replicas"]

    if "n_replicas" in params.keys():
      self._n_replicas = params["n_replicas"]

    if "is_chief" in params.keys():
      self._is_chief = params["is_chief"]

    self._batch_size_shard = self._batch_size // self._n_gpu

  def _init_optimizer(self):
    """
    initialize the optimizer.
    [TODO] expose the optimizer type
    """
    lr = tf.constant(self._learning_rate)
    self._optimizer = tf.train.AdamOptimizer(learning_rate=lr)

  @property
  def global_step(self):
    return self._global_step

  @global_step.setter
  def global_step(self, value):
    self._global_step = value

  @abc.abstractmethod
  def _network(self, input):
    """
    build the netwrok layers.
    """
    raise NotImplementedError


  @abc.abstractmethod
  def _loss(self, _data):
    """
    evaluate the network loss.
    """
    raise NotImplementedError


  def _gradient(self, loss_op):
    """
    evaluate the network variable gradients.
    """
    assert loss_op is not None, "initialize the network loss operation !"
    assert self._optimizer is not None, "initialize the network optimizer !"

    _var_collection = tf.get_collection(
        key= tf.GraphKeys.TRAINABLE_VARIABLES,
        scope= self._var_scope)

    _gradient_op = self._optimizer.compute_gradients(
        loss= loss_op,
        var_list= _var_collection)

    return _gradient_op


  def build_forward(self, input, reuse= False):
    """
    build the network forward operation.
    """
    with tf.variable_scope(self._var_scope, reuse= reuse):
      forward_op = self._network(input = input)

    return forward_op


  def build_device_loss(self, data, reuse= False):
    """
    build the loss and gradient per device.
    """
    with tf.variable_scope(self._var_scope, reuse= reuse):
      loss_op = self._loss(data)
      gradient_op = self._gradient(loss_op)

    return loss_op, gradient_op


  def build_host_train(self, avg_gradients, reuse= False):
    """
    build the train operation on the host.
    """
    hook_sync_replica = None
    optimizer = self._optimizer

    LOGGER.debug('REUSE {}'.format(reuse))
    with tf.variable_scope("host_optimizer", reuse= tf.AUTO_REUSE):
      if self._sync_replicas:
        optimizer = tf.train.SyncReplicasOptimizer(
            opt= self._optimizer,
            replicas_to_aggregate= self._n_replicas)
        hook_sync_replica = optimizer.make_session_run_hook(is_chief= self._is_chief)

      # train_op = optimizer.apply_gradients(avg_gradients, global_step=tf.train.get_global_step())
      train_op = optimizer.apply_gradients(avg_gradients, global_step= self.global_step)

    ## LOGGER.DEBUG_VARIABLES()

    return train_op, hook_sync_replica


  @staticmethod
  def compute_avg_gradients(device_gradients):
    """
    [source avg. proc] - https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/
    """
    # list of tuples [(grad_0_avg, var_0), (grad_1_avg, var_1), ...]
    avg_gradients = list()

    # LOGGER.debug('DEVICE GRADIENTS ----- {}'.format(device_gradients))

    # grad_var_tuples ((grad_0_device_0, var_0_device_0), (grad_0_device_1, var_0_device_1), ...)
    for grad_var_tuples in zip(*device_gradients):
      grads = []

      for g, _ in grad_var_tuples:
        # add a device dimension to average over.
        if g is not None:
          expanded_g = tf.expand_dims(g, 0)
          grads.append(expanded_g)

      if len(grads) == 0:
        continue
      # Average over the 'device' dimension.
      grad = tf.concat(axis= 0, values= grads)
      grad = tf.reduce_mean(grad, 0)

      v = grad_var_tuples[0][1]
      grad_and_var = (grad, v)
      avg_gradients.append(grad_and_var)

    return avg_gradients


#-------------------------
# __placeholder__ 
#-------------------------
class Encoder(Network):
  pass

class Decoder(Network):
  pass

class EncoderD(Network):
  pass

class DecoderD(Network):
  pass

