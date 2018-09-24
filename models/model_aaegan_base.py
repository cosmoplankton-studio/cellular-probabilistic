#
# ---------------------------------------------------------
# Copyright 2018-present (c) Automatos Studios. All Rights Reserved.
# ---------------------------------------------------------
# 
# @file       model_aaegan_base.py
# @author     cosmoplankton@automatos.studio
#


"""
  #===================================================
  # The Model Implementation. This targets the tensorflow library.
  #===================================================

  It implements the training algorithm step, model export and checkpoint save/load.
  Training algorithm: Algorithm 01/02 - Training procedure reference model.
  @reference: Building a 3D Integrated Cell - https://doi.org/10.1101/238378
"""


import copy
from types import SimpleNamespace

import abc
import numpy as np
import tensorflow as tf

from networks import network_aae_base as networks
from models import model_defines as md

from utils import global_logging as logging
LOGGER = logging.get_logger()



class Model(abc.ABC):
  """
  It implements the training algorithm step, model export and checkpoint save/load.
  Training algorithm: Algorithm 01/02 - Training procedure reference structure model.
  Building a 3D Integrated Cell - https://doi.org/10.1101/238378
  ===========================================
  Args:
    params: Multilevel dictionary containing the initialization and run parameters.
      1. dataset: parameters specific to DATA_CONNECTIONS to the model.
      2. network: parameters specific to architecture of component networks.
      3. compute: parameters specific to computing resources.
      4. train: parameters specific to training run.
  """

  def __init__(self, dataservice, params):
    """
    Args:
      params: Multilevel dictionary containing the initialization and run parameters.

    Raises:
      ValueError:
      1. If parameters are invalid.
    """

    self._dataservice = dataservice
    self._options = {
        'dataset': SimpleNamespace(),
        'network': SimpleNamespace(),
        'compute': SimpleNamespace(),
        'train': SimpleNamespace()
    }

    if not self._init_params(params):
      raise ValueError('invalid model initialization parameters. check log for details')

    self._hooks = {
        md.Hooks.TRAIN: list(),
        md.Hooks.SUMMARY: list(),
        md.Hooks.LOG: list(),
        md.Hooks.CHECKPOINT: list()
    }

    self._export = {
        'encoder_inp': None,
        'encoder_out': None,
        'decoder_inp': None,
        'decoder_out': None
    }


  def _init_params(self, params):
    """
    Validate and unpack the model initialization parameters
    """
    try:
      self._options['compute'].n_gpu = params.n_gpu
      self._options['compute'].sync_replicas = params.sync_replicas
      self._options['compute'].n_replica_workers = params.n_replica_workers
      self._options['compute'].is_chief = params.is_chief
      self._options['compute'].param_device = params.param_device
      self._options['compute'].worker_device = params.worker_device

      self._options['dataset'].mode = params.run_mode
      self._options['dataset'].img_pipeline = params.img_pipeline
      self._options['dataset'].batch_size = params.batch_size
      self._options['dataset'].batch_size_shard = params.batch_size / params.n_gpu
      self._options['dataset'].n_epochs = params.n_epochs
      self._options['dataset'].n_max_iters = params.n_max_iters
      self._options['dataset'].n_iters_per_epoch = params.n_iters_per_epoch

      self._options['network'].n_latent_dim = params.n_latent_dim
      self._options['network'].n_classes = params.n_classes

      self._options['train'].lrate_encoder = params.lrate_encoder
      self._options['train'].lrate_decoder = params.lrate_decoder
      self._options['train'].lrate_encoder_d = params.lrate_encoder_d
      self._options['train'].lrate_decoder_d = params.lrate_decoder_d
      self._options['train'].lambda_encoder_d = params.lambda_encoder_d
      self._options['train'].lambda_decoder_d = params.lambda_decoder_d

    except AttributeError:
      LOGGER.error("Incorrect Model initialization parameter.")
      raise

    return True

  @abc.abstractmethod
  def setup(self):
    """
    Setup the model component networks.
    IMPORTANT: Appropriately override the 'networks' creation.
    """
    params = dict()

    # common parameters
    params['n_gpu'] = self._options['compute'].n_gpu
    params['sync_replicas'] = self._options['compute'].sync_replicas
    params['n_replicas'] = self._options['compute'].n_replica_workers
    params['is_chef'] = self._options['compute'].is_chief
    params['reuse'] = False
    params['latent_dim'] = self._options['network'].n_latent_dim

    # encoder_d
    params['lrate'] = self._options['train'].lrate_encoder_d
    params['batch_size'] = self._options['dataset'].batch_size
    self._encoder_d = networks.EncoderD(self, params=params)

    # decoder_d
    params['lrate'] = self._options['train'].lrate_decoder_d
    self._decoder_d = networks.DecoderD(self, params=params)

    # encoder
    params['lrate'] = self._options['train'].lrate_encoder
    params['lambda_d'] = self._options['train'].lambda_encoder_d
    self._encoder = networks.Encoder(self, params=params)

    # decoder
    params['lrate'] = self._options['train'].lrate_decoder
    params['lambda_d'] = self._options['train'].lambda_decoder_d
    self._decoder = networks.Decoder(self, params=params)


  def build_model(self):
    """
    Build the Model Training Operation (All Component Networks)
    Training Sequence overview (enforced using tf.control_dependencies()):
      encoder_d >> decoder_d >> encoder >> decoder
    """
    device_data = dict()
    for _device_id in range(0, self._options['compute'].n_gpu):
      device_data[_device_id] = dict()

    with tf.device('/{}:0'.format(self._options['compute'].param_device)):
      # create different global steps for component networks for distributed synchronous learning.
      self._encoder_d.global_step = Model._create_optim_step(name= 'global_step_enc_d')
      self._decoder_d.global_step = Model._create_optim_step(name= 'global_step_dec_d')
      self._encoder.global_step = Model._create_optim_step(name= 'global_step_enc')
      self._decoder.global_step = Model._create_optim_step(name= 'global_step_dec')

      # build the component networks (for training).
      LOGGER.set_mode(logging.LogMode.OFF)
      LOGGER.debug('_build_encoder_d')

      loss_enc_d, train_enc_d = self._build_encoder_d(device_data)

      LOGGER.debug('_build_decoder_d')

      with tf.control_dependencies([loss_enc_d, train_enc_d]):
        loss_dec_d, train_dec_d = self._build_decoder_d(device_data)

        LOGGER.debug('_build_encoder')

        with tf.control_dependencies([loss_dec_d, train_dec_d]):
          loss_enc, train_enc = self._build_encoder(device_data)

          LOGGER.debug('_build_decoder')

          with tf.control_dependencies([loss_enc, train_enc]):
            loss_dec, train_dec = self._build_decoder(device_data)

    # group the component networks' training operations.
    model_train_op = tf.group([train_enc_d, train_dec_d, train_enc, train_dec])

    # [DEBUG] // messages START
    # LOGGER.set_mode(logging.LogMode.TF_LOGGING)
    LOGGER.DEBUG_VARIABLES()
    LOGGER.DEBUG_LOSSES()
    # [DEBUG] // messages END

    optim_global_steps = [
        self._encoder_d.global_step,
        self._decoder_d.global_step,
        self._encoder.global_step,
        self._decoder.global_step]

    return model_train_op, optim_global_steps


  def _build_encoder_d(self, device_data):
    """
    Build Encoder Discriminator Training Operation.

    Args:
      device_data:  Per device data outputs of previous networks.
              This is used to create the graph per device and use output pf one network
              as input for other during the same iteration.

      device_data['device_index']['tensor_name']
    """
    device_losses = list()
    device_gradients = list()

    for pin_id in range(0, self._options['compute'].n_gpu):
      data = dict()
      reuse = bool(pin_id != 0)
      device_str = self._options['compute'].worker_device

      with tf.device('/{0}:{1}'.format(device_str, pin_id)):
        # get the images
        data_pipeline = self._dataservice.get_data_pipeline(
            self._options['dataset'].img_pipeline)

        device_data[pin_id]['x_real'] = data_pipeline.get_dataset_tfrecords(device_idx = pin_id)

        # get the normal distribution
        device_data[pin_id]['z_real'] = self._dataservice.get_normal_dist(
            sigma_std= 0.01,
            element_shape= [self._options['network'].n_latent_dim],
            device_idx= pin_id)

        device_data[pin_id]['z_fake'] = self._encoder.build_forward(
            device_data[pin_id]['x_real'],
            reuse = reuse)

        data['y_Hat_z_real'] = self._encoder_d.build_forward(
            input= device_data[pin_id]['z_real'],
            reuse= reuse)

        data['y_Hat_z_fake'] = self._encoder_d.build_forward(
            input= device_data[pin_id]['z_fake'],
            reuse= True)

        loss, gradient = self._encoder_d.build_device_loss(data= data, reuse= reuse)
        device_losses.append(loss)
        device_gradients.append(gradient)

        # add summary for 'x_real'
        if pin_id == 0 and self._options['compute'].is_chief:
          self._add_image_summary('x_real', device_data[pin_id]['x_real'])

        # prepare the tensors for serving export
        if not reuse:
          self._encoder_inp = device_data[pin_id]['x_real']
          self._encoder_out = device_data[pin_id]['z_fake']
          self._decoder_inp = device_data[pin_id]['z_real']

    # average the gradients from GPUs
    avg_gradient = self._encoder_d.compute_avg_gradients(device_gradients)

    train_op, hook_sync_replica = self._encoder_d.build_host_train(avg_gradient, reuse= False)
    loss_op = tf.reduce_mean(device_losses, name='loss_enc_d')

    if hook_sync_replica:
      self._hooks[md.Hooks.TRAIN].append(hook_sync_replica)

    # add summary hooks
    loss_summary = tf.summary.scalar(name= 'loss_enc_d', tensor= loss_op)
    hook_summary = tf.train.SummarySaverHook(summary_op=loss_summary, save_steps=1)
    self._hooks[md.Hooks.SUMMARY].append(hook_summary)

    return loss_op, train_op


  def _build_decoder_d(self, device_data):
    """
    Build Decoder Discriminator Training Operation.

    Args:
      device_data:  Per device data outputs of previous networks.
              This is used to create the graph per device and use output pf one network
              as input for other during the same iteration.

      device_data['device_index']['tensor_name']
    """
    device_losses = list()
    device_gradients = list()

    for pin_id in range(0, self._options['compute'].n_gpu):
      data = dict()
      reuse = bool(pin_id != 0)
      device_str = self._options['compute'].worker_device

      with tf.device('/{0}:{1}'.format(device_str, pin_id)):
        device_data[pin_id]['x_fake'] = self._decoder.build_forward(
          input= device_data[pin_id]['z_fake'],
          reuse= reuse)

        device_data[pin_id]['x_decode'] = self._decoder.build_forward(
          input= device_data[pin_id]['z_real'],
          reuse= True)

        data['y_Hat_x_real'] = self._decoder_d.build_forward(
          input= device_data[pin_id]['x_real'],
          reuse = reuse)

        data['y_Hat_x_fake'] = self._decoder_d.build_forward(
          input= device_data[pin_id]['x_fake'],
          reuse= True)

        data['y_Hat_x_decode'] = self._decoder_d.build_forward(
          input=device_data[pin_id]['x_decode'],
          reuse= True)

        loss, gradient = self._decoder_d.build_device_loss(data= data, reuse = reuse)
        device_losses.append(loss)
        device_gradients.append(gradient)

        # add summary for 'x_fake'
        if pin_id == 0 and self._options['compute'].is_chief:
          self._add_image_summary('x_fake', device_data[pin_id]['x_fake'])

        # add summary for 'x_decode'
        if pin_id == 0 and self._options['compute'].is_chief:
          self._add_image_summary('x_decode', device_data[pin_id]['x_decode'])

        # prepare the tensors for serving export
        if not reuse:
          self._decoder_out = device_data[pin_id]['x_decode']

    avg_gradient = self._decoder_d.compute_avg_gradients(device_gradients)

    train_op, hook_sync_replica = self._decoder_d.build_host_train(avg_gradient, reuse=True)
    loss_op = tf.reduce_mean(device_losses, name='loss_dec_d')

    # add training hooks
    if hook_sync_replica:
      self._hooks[md.Hooks.TRAIN].append(hook_sync_replica)

    # add summary hooks
    loss_summary = tf.summary.scalar(name= 'loss_dec_d', tensor= loss_op)
    hook_summary = tf.train.SummarySaverHook(summary_op=loss_summary, save_steps=1)
    self._hooks[md.Hooks.SUMMARY].append(hook_summary)

    return loss_op, train_op


  def _build_encoder(self, device_data):
    """
    Build Encoder Training Operation.

    Args:
      device_data:  Per device data outputs of previous networks.
              This is used to create the graph per device and use output pf one network
              as input for other during the same iteration.

      device_data['device_index']['tensor_name']
    """
    device_losses = list()
    device_gradients = list()

    for pin_id in range(0, self._options['compute'].n_gpu):
      data = dict()
      reuse = bool(pin_id != 0)
      device_str = self._options['compute'].worker_device

      with tf.device('/{0}:{1}'.format(device_str, pin_id)):
        cross_ent_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels= device_data[pin_id]['x_real'],
          logits= device_data[pin_id]['x_fake'])

        data['x_recons_error'] = tf.reduce_mean(cross_ent_loss)

        data['y_Hat_z_fake'] = self._encoder_d.build_forward(
          input= device_data[pin_id]['z_fake'],
          reuse= True)

        loss, gradient = self._encoder.build_device_loss(data= data, reuse= reuse)
        device_losses.append(loss)
        device_gradients.append(gradient)

        device_data[pin_id]['x_recons_error'] = data['x_recons_error']

    avg_gradient = self._encoder.compute_avg_gradients(device_gradients)

    train_op, hook_sync_replica = self._encoder.build_host_train(avg_gradient, reuse= True)
    loss_op = tf.reduce_mean(device_losses, name='loss_enc')

    if hook_sync_replica:
      self._hooks[md.Hooks.TRAIN].append(hook_sync_replica)

    # add summary hooks
    loss_summary = tf.summary.scalar(name= 'loss_enc', tensor= loss_op)
    hook_summary = tf.train.SummarySaverHook(summary_op=loss_summary, save_steps=1)
    self._hooks[md.Hooks.SUMMARY].append(hook_summary)

    return loss_op, train_op


  def _build_decoder(self, device_data):
    """
    Build Decoder Training Operation.

    Args:
      device_data:  Per device data outputs of previous networks.
              This is used to create the graph per device and use output pf one network
              as input for other during the same iteration.

      device_data['device_index']['tensor_name']
    """
    device_losses = list()
    device_gradients = list()

    for pin_id in range(0, self._options['compute'].n_gpu):
      data = dict()
      reuse = bool(pin_id != 0)
      device_str = self._options['compute'].worker_device

      with tf.device('/{0}:{1}'.format(device_str, pin_id)):
        data['y_Hat_x_fake'] = self._decoder_d.build_forward(
          input= device_data[pin_id]['x_fake'],
          reuse= True)

        data['y_Hat_x_decode'] = self._decoder_d.build_forward(
          input= device_data[pin_id]['x_decode'],
          reuse= True)

        data['x_recons_error'] = device_data[pin_id]['x_recons_error']

        loss, gradient = self._decoder.build_device_loss(data= data, reuse= reuse)
        device_losses.append(loss)
        device_gradients.append(gradient)

    avg_gradient = self._decoder.compute_avg_gradients(device_gradients)
    train_op, hook_sync_replica = self._decoder.build_host_train(avg_gradient, reuse= True)
    loss_op = tf.reduce_mean(device_losses, name= 'loss_dec')

    if hook_sync_replica:
      self._hooks[md.Hooks.TRAIN].append(hook_sync_replica)

    # add summary hooks
    loss_summary = tf.summary.scalar(name= 'loss_dec', tensor= loss_op)
    hook_summary = tf.train.SummarySaverHook(summary_op=loss_summary, save_steps=1)
    self._hooks[md.Hooks.SUMMARY].append(hook_summary)

    return loss_op, train_op

  @property
  def _encoder_inp(self):
    return self._export['encoder_inp']

  @_encoder_inp.setter
  def _encoder_inp(self, tensor):
    self._export['encoder_inp'] = tf.identity(tensor, name= 'export_encoder_inp')

  @property
  def _encoder_out(self):
    return self._export['encoder_out']

  @_encoder_out.setter
  def _encoder_out(self, tensor):
    self._export['encoder_out'] = tf.identity(tensor, name= 'export_encoder_out')

  @property
  def _decoder_inp(self):
    return self._export['decoder_inp']

  @_decoder_inp.setter
  def _decoder_inp(self, tensor):
    self._export['decoder_inp'] = tf.identity(tensor, name= 'export_decoder_inp')

  @property
  def _decoder_out(self):
    return self._export['decoder_out']

  @_decoder_out.setter
  def _decoder_out(self, tensor):
    self._export['decoder_out'] = tf.identity(tensor, name= 'export_decoder_out')


  def export_serving(self, sess, export_dir, as_text= True):
    """
    * Exports the model for serving.
    * Exports the entire graph.
    * Device info is cleared for serving export.
    * Exported models serve as payloads for both standard tf.ModelServer and AWS lambda backend. 
    [TODO] Export only the encoder and decoder, not the discriminators.
    """
    tf.add_to_collection('encoder_inp', self._encoder_inp)
    tf.add_to_collection('encoder_out', self._encoder_out)
    tf.add_to_collection('decoder_inp', self._decoder_inp)
    tf.add_to_collection('decoder_out', self._decoder_out)

    saver = tf.saved_model.builder.SavedModelBuilder(export_dir= export_dir)

    tensor_info_enc_inp = tf.saved_model.utils.build_tensor_info(self._encoder_inp)
    tensor_info_enc_out = tf.saved_model.utils.build_tensor_info(self._encoder_out)

    tensor_info_dec_inp = tf.saved_model.utils.build_tensor_info(self._decoder_inp)
    tensor_info_dec_out = tf.saved_model.utils.build_tensor_info(self._decoder_out)

    enc_generate_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs= {'image': tensor_info_enc_inp},
        outputs= {'latent_rep': tensor_info_enc_out},
        method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    dec_generate_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs= {'random_key': tensor_info_dec_inp},
        outputs= {'image': tensor_info_dec_out},
        method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    saver.add_meta_graph_and_variables(
        sess= sess,
        tags= [tf.saved_model.tag_constants.SERVING],
        signature_def_map= {
            'encode': enc_generate_signature,
            'decode': dec_generate_signature
        },
        main_op= None,
        assets_collection= None,
        clear_devices= True)

    saver.save(as_text= as_text)

  def save_checkpoint(self, **kwargs):
    """ Not needed as using MonitoredTrainingSession """
    raise NotImplementedError

  def load_checkpoint(self, **kwargs):
    """ Not needed as using MonitoredTrainingSession """
    raise NotImplementedError


  def get_hooks(self, hook_type):
    """
    supported hook types:
      md.Hooks.CHECKPOINT
      md.Hooks.LOG
      md.Hooks.SUMMARY
      md.Hooks.TRAIN
    """
    if hook_type in self._hooks.keys():
      return self._hooks[hook_type]

    raise ValueError('unsupported hook type {}'.format(hook_type))


  def _add_image_summary(self, image_tag, image):
    """
    add slices of 3d-images. tf.summary.image only handles 2d images.
    """
    num_slice = 5
    # image shape assumes channel_last [batch, x, y, z, chnl]
    image_x_pixels = image.shape[1]
    stride = image_x_pixels // (num_slice-1)

    for i in range(num_slice):
      index = i*stride - 1 if i else i*stride
      image_slice = image[:,index,:, :, :]
      summary_images = tf.summary.image(name= '{0}_{1}'.format(image_tag, i), tensor= image_slice, max_outputs =1)
      summary_hook = tf.train.SummarySaverHook(summary_op= summary_images, save_steps= 1)
      self._hooks[md.Hooks.SUMMARY].append(summary_hook)

  @staticmethod
  def _create_optim_step(name):
    """
    Create separate steps for each optimizer.
    """
    step = tf.get_variable(
        name= name,
        shape=[],
        initializer= tf.constant_initializer(1),
        trainable= False,
        dtype= tf.int64)

    return step
