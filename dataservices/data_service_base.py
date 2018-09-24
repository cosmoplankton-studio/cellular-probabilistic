#
# ---------------------------------------------------------
# Copyright 2018-present (c) Automatos Studios. All Rights Reserved.
# ---------------------------------------------------------
# 
# @file       data_service_base.py
# @author     cosmoplankton@automatos.studio
#


"""
Implementation of the Dataservice Base.
* A Model uses a single Service.
* A Service contains multiple pipelines (for most cases it contains only one).
"""


import os
import sys
import abc
import pickle
import tarfile

from urllib.request import urlopen

import numpy as np
import tensorflow as tf

from dataservices import data_pipeline_base as dpipe

from utils import global_logging as logging
LOGGER = logging.get_logger()


class ServiceBase:
  """
  Base Service.
  A Model uses a single Service.
  A Service contains multiple pipelines (for most cases it contains only one).
  """

  def __init__(self):
    self._copy_remote = False
    self._pipelines = dict()

  def setup(self, params):
    self._init_params(params)


  def _init_params(self, params):
    """
    initialize the parameters.
    """
    try:
      self._n_gpu = params.n_gpu
      self._batch_size = params.batch_size
      self._iters_per_epoch = params.n_iters_per_epoch
      self._max_iters = params.n_max_iters
      self._replica_index = params.task_idx
      self._n_epocs = params.n_epochs
    except AttributeError:
      LOGGER.error("--- Incorrect dataservice init parameters. see log for details ---")
      LOGGER.error(sys.exc_info()[0])
      raise

    if(self._max_iters % self._iters_per_epoch != 0):
      LOGGER.error("--- max_iters not a multiple of iters_per_epoch ---")
      exit(-1)

    self._batch_size_shard = self._batch_size // self._n_gpu

  def get_data_pipeline(self, KEY):
    """
    Get the existing data-pipelines.
    """
    if KEY not in self._pipelines.keys():
      raise ValueError("pipeline with this KEY does not exists")

    return self._pipelines[KEY]


  def create_data_pipeline(self, params, tag= 'debug_data'):
    """
    create the various dataset pipelines available.
    Args:
      tag: pipeline type.
      params:
        params.data_file: data file name.
        params.data_dir: data directory.
        params.data_url: data url.
    """
    _pipeline_idx = 0
    _pipeline_key = None

    tag = tag.lower()

    if(tag == 'debug_data'):
      _pipeline_key = 'debug_data_{0}'.format(_pipeline_idx)
      self._pipelines[_pipeline_key] = dpipe.PipelineDebug(self, params)
    else:
      raise NotImplementedError

    return _pipeline_key


  def get_normal_dist(self, mean_mu = 0.0, sigma_std = 0.01, element_shape=[], device_idx = 0):
    """
    Sample from a normal/gaussian distribution.
    """
    # It is important that we always use a placeholder for batch_size.
    # Using a placeholder keeps the batch_size from being baked into the graph.
    # The batch_size getting baked into the graph is problematic for model for serving.

    batch_size = tf.placeholder_with_default(input= tf.constant(self._batch_size_shard, shape =[], dtype= tf.int64), shape= [])
    tf.add_to_collection('batch_size_placeholder', batch_size)

    sample_shape = [batch_size]
    sample_shape.extend(element_shape)

    sample_norm = tf.random_normal(shape= sample_shape, mean= mean_mu, stddev= sigma_std)

    return sample_norm


  def get_binary_data_from_files(self, filenames, record_bytes, record_shape= [256, 256, 3], dt= tf.uint8):
    """
    raises:
      ValueError - If record_shape is not compatible with actual data.
    """

    def _parse(record):
      """
      parse and filter the binary data.
      """
      record_data = tf.decode_raw(record, dt)
      record_data.set_shape([np.prod(record_shape)])
      record_data = tf.reshape(record_data, shape= record_shape)
      record_data = tf.cast(record_data, tf.float32)

      return record_data

    dataset = tf.data.FixedLengthRecordDataset(filenames= filenames,
                                               record_bytes= record_bytes)


    dataset = dataset.repeat(self._n_epocs)
    dataset = dataset.map(_parse)
    dataset = dataset.batch(self._batch_size)
    next_element = dataset.make_one_shot_iterator().get_next()

    return next_element


  def get_image_from_tfrecords(self, filenames, img_shape=[256, 256, 256, 1], dt=tf.uint8):
    """
    Args:
      img_shape= image shape in channel_last notation. e.g. [256, 256, 256, 1]
    """

    def _parse_record(record):
      """
      parse 3d-image data from the tf.Example proto
      """
      img_data_desc = { "voxels" : tf.FixedLenFeature(shape=[], dtype=tf.string) }
      img_data = tf.parse_single_example(record, features=img_data_desc)

      image = tf.decode_raw(img_data["voxels"], dt)
      image.set_shape([np.prod(img_shape)])
      image = tf.reshape(image,shape=img_shape)
      image = tf.cast(image, tf.float32)

      LOGGER.debug('--- img_shape in after parse_records: {} ---'.format(image.get_shape()))

      return image

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.repeat(self._n_epocs)
    dataset = dataset.map(_parse_record)
    dataset = dataset.batch(self._batch_size)

    LOGGER.debug('--- dataset shape after batch: {} ---'.format(dataset.output_shapes))

    next_element = dataset.make_one_shot_iterator().get_next()

    return next_element


  @staticmethod
  def store_np_img_as_tfrecords(np_imgs, out_filename):
    """
    Args:
      np_imgs: list of images represented as np array.
      out_filename: output file.
    """
    with tf.python_io.TFRecordWriter(out_filename) as writer:
      for img in np_imgs:
        example_proto = tf.train.Example(features=tf.train.Features(
            feature={'voxels':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()]))}
            ))
        writer.write(record=example_proto.SerializeToString())


  @staticmethod
  def parse_jpeg_file(filename):
    """
    Args:
      filename: .jpeg file
    Returns:
      uint8 tf.Tensor
    """
    image_raw = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_raw, channels=0)
    image_resized = tf.image.resize_images(image_decoded, size=[64, 64], method=tf.image.ResizeMethod.BILINEAR)

    return image_resized

  @staticmethod
  def extract_tarball(src, target):
    """
    Extract an archive.
    """
    file = tarfile.open(src)
    try:
      os.mkdir(target)
    except:
      raise ValueError("Archive extract failure. Check file paths/permissions")

    file.extractall(target)
    file.close()


  @staticmethod
  def download_url(src_url, target_file):
    """
    Download the source url to the target_file.
    """
    if not os.path.exists(target_file):
      handle_url = urlopen(src_url)
      handle_file = open(target_file, 'wb')
      handle_file.write(handle_url.read())
    else:
      raise FileExistsError('Download failuere. File already exists, provide unique target path')


  @staticmethod
  def import_mri_to_np(folder_path, file_prefix, num_slices, shape_slice= [256, 256], dt='>u2'):
    """
    assumptions: a. file naming convention file_prefix.[index]
    """
    img_data = np.array([np.fromfile(os.path.join(folder_path, '{1}.{2}'.format(file_prefix, idx)),
                     dtype=dt)
               for idx in range(1, num_slices + 1)])

    img_shape = [num_slices]
    img_shape.extend(shape_slice)
    img_data = img_data.reshape(img_shape)

    return img_data


  @staticmethod
  def import_pickle_data(filename):
    """
    read python pickeled data
    [source] img-dataset-storage- http://www.cs.toronto.edu/~kriz/cifar.html
    """
    data = None
    try:
      hanlde = open(filename, 'rb')
      data = pickle.load(handle, encoding='BYTES')
      hanlde.close()
    except IOError:
      pass

    return data