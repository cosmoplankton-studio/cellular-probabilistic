#
# ---------------------------------------------------------
# Copyright 2018-present (c) Automatos Studios. All Rights Reserved.
# ---------------------------------------------------------
# 
# @file       data_pipeline_base.py
# @author     cosmoplankton@automatos.studio
#

"""
Implementaion of the Data PipelineBase and PipelineDebug.
A single Dataservice contains multiple pipelines.
"""


import os
import sys
import abc

import numpy as np
import tensorflow as tf

import json

from utils import global_logging as logging
LOGGER = logging.get_logger()



class PipelineType:
  DEBUG = 'debug_data'



class PipelineBase(abc.ABC):
  """
  Base class for data pipelines.
  A dataservice could have multiple pipelines feeding multiple networks.
  """
  def __init__(self, service, params):
    try:
      self._service = service
      self._data_dir = params.data_dir
      self._data_file = params.data_file
      self._data_src = params.data_src
    except AttributeError:
      LOGGER.error("improper parameters. check log for details")
      raise

  @abc.abstractmethod
  def get_max_dataitems(self):
    raise NotImplementedError

  @abc.abstractmethod
  def get_dataset_tfrecords(self, device_idx = 0):
    raise NotImplementedError



class PipelineDebug(PipelineBase):
  """
  a.  Debug pipeline. creates dataset to test the training operations/network.
      This doesn't test performance or convergence or behavior related to actual datasets..
  b.  Create batches of primitive 3-d point cloud.
  """
  def __init__(self, service, params):
    super().__init__(service, params)

  def get_max_dataitems(self):
    """
    Return:
      None: infinite dataset.
    """
    # [IMP] Read it from the metadata json file.
    return 20

  def get_dataset_tfrecords(self, device_idx = 0):
    """
    load the tfrecord datasets
    """
    tf_records = os.path.join(self._data_dir, self._data_file)
    f_metadata = open(tf_records + '.metadata', 'r', encoding= 'utf-8')
    metadata = json.load(f_metadata)
    img_shape = metadata['img_shape']
    dtype = 'tf.' + metadata['d_type']
    f_metadata.close()

    data_iter = self._service.get_image_from_tfrecords(
        filenames= [tf_records],
        img_shape= img_shape,
        dt= eval(dtype))

    LOGGER.debug('----- TFRECORDS DATA ITER {} -----'.format(data_iter))

    return data_iter

