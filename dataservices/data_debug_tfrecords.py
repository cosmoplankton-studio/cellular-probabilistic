#
# ---------------------------------------------------------
# Copyright 2018-present (c) Automatos Studios. All Rights Reserved.
# ---------------------------------------------------------
# 
# @file       data_debug_tfrecords.py
# @author     cosmoplankton@automatos.studio
#

"""
Implementation of the DebugData.
* Provides utilities for generation of debug and synthetic lite-data.
* Stores data in tfrecord format.
"""


import argparse
import os
import sys

import numpy as np
import tensorflow as tf

import json

sys.path.append(os.getcwd())

from utils import global_logging as logging
LOGGER = logging.get_logger()

OPTIONS = None



class DebugData:
  def __init__(self):
    pass

  @staticmethod
  def create_debug_dataset(data_file, n_dataitems= 20, img_dim= 256, cube_edge= 8, n_channels= 1):
    images = list()
    for i in range(n_dataitems):
      images.append(DebugData.get_point_cloud_cube(
          img_dim= img_dim,
          cube_edge= cube_edge,
          n_channels= n_channels))

    DebugData.store_np_img_as_tfrecords(images, data_file)


  @staticmethod
  def store_np_img_as_tfrecords(np_imgs, out_filename):
    """
    Args:
      np_imgs: list of images represented as np array.
      out_filename: output file.
    """
    img_shape = None
    with tf.python_io.TFRecordWriter(out_filename) as writer:
      for img in np_imgs:
        # write .tfrecords file
        example_proto = tf.train.Example(features= tf.train.Features(
            feature={'voxels':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()]))}
            ))
        writer.write(record=example_proto.SerializeToString())

        if img_shape is None:
          img_shape = img.shape

    # write .metadata file
    metadata = dict()
    metadata['img_shape'] = img_shape
    metadata['d_type'] = 'uint8'

    fp_metadata = open(out_filename + '.metadata', 'w', encoding= 'utf-8')
    json.dump(metadata, fp_metadata)
    fp_metadata.close()


  @staticmethod
  def get_point_cloud_cube(img_dim= 256, cube_edge= 8, n_channels= 1):
    """
    creates samples of 3d-point clouds representing a cube.
    """
    img_voxels = np.zeros(shape=(img_dim, img_dim, img_dim, n_channels), dtype=np.uint8)
    padding = (img_dim - cube_edge) // 2
    extent = padding + cube_edge
    img_voxels[padding:extent, padding:extent, padding:extent, :] = 1

    return img_voxels


def prepare_cmd_args(parser = None):
  if parser is None:
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='create .tfrecords for large numpy array datasets.')

  parser.add_argument('--data_dir', type=str, default="debug_data", help='debug data directory')
  parser.add_argument('--data_file', type=str, default="point_cloud_256px_dataitems_20_uint8.tfrecord", help='data file')
  parser.add_argument('--n_dataitems', type=int, default=20, help='')
  parser.add_argument('--img_dim', type=int, default=256, help='')
  parser.add_argument('--shape_dia', type=int, default=32, help='')
  parser.add_argument('--n_channels', type=int, default=1, help='n0. of channels')

  return parser.parse_known_args()



if __name__ == "__main__":
  try:
    # preapare the options
    OPTIONS, unknown_options = prepare_cmd_args()
    LOGGER.message('...PARSED KNOWN ARGUMENTS...')
    LOGGER.message(OPTIONS)

    # prepare data directories and files
    if not os.path.exists(OPTIONS.data_dir):
      os.mkdir(OPTIONS.data_dir)

    OPTIONS.data_file = os.path.join(OPTIONS.data_dir, OPTIONS.data_file)

    DebugData.create_debug_dataset(data_file= OPTIONS.data_file,
                                   n_dataitems= OPTIONS.n_dataitems,
                                   img_dim= OPTIONS.img_dim,
                                   cube_edge= OPTIONS.shape_dia,
                                   n_channels= OPTIONS.n_channels)

  except:
    LOGGER.error(".... ERROR DURING DATA EXPORT ....")
    LOGGER.error(sys.exc_info()[0])
    raise