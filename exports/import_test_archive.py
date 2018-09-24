#
# ---------------------------------------------------------
# Copyright 2018-present (c) Automatos Studios. All Rights Reserved.
# ---------------------------------------------------------
# 
# @file       import_test_archive.py
# @author     cosmoplankton@automatos.studio
#

"""
Import model >> Test on a single sample >> Archive if no error.
"""

import argparse
import os
import sys
import shutil

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import json

sys.path.append(os.getcwd())

from utils import global_logging as logging
LOGGER = logging.get_logger()

OPTIONS = None


def import_test_archive_model(dir):
  """
  Import model >> Test on a single sample >> Archive if no error.
  """
  if OPTIONS.export_tag.lower() == 'serving':
    tags = [tf.saved_model.tag_constants.SERVING]
  else:
    LOGGER.error('Only \'serving\' [--export_tag] is allowed.')
    sys.exit(-1)

  with tf.Session() as sess:
    # load the exported graph
    tf.saved_model.loader.load(sess= sess, tags= tags, export_dir= dir)

    # generate decoder output
    out_dec = _run_decoder(sess, dir)
    _generate_slices(sess, dir, out_dec)

    # generate encoder output
    # out_enc = _run_encoder(sess, dir)
    # _generate_json(sess, dir, out_enc)

    # archive model and test outputs
    if OPTIONS.save_archive:
      _archive_model_and_test_output(dir)

def _run_decoder(sess, dir):
  """
  Run to generate the encoder output.
  """
  decoder_inp = tf.get_collection('decoder_inp')[0]
  decoder_out = tf.get_collection('decoder_out')[0]
  batch_size_pl = tf.get_collection('batch_size_placeholder')[0]

  LOGGER.debug(decoder_inp.shape)
  LOGGER.debug(decoder_out.shape)

  dec_inp = np.zeros(shape= (1, OPTIONS.n_latent_dim))
  num_inp = 1

  feed_dict = {
      decoder_inp.name: dec_inp,
      batch_size_pl.name: num_inp
  }

  generated_img = sess.run(decoder_out, feed_dict= feed_dict)

  return generated_img


def _generate_slices(sess, dir, img_3d, num_slices= 5):
  """
  Generate slices from 3d-image and save themm as 'png' files
  """
  slices = []
  image_x_pixels = OPTIONS.img_shape[0]
  stride = image_x_pixels // (num_slices-1)

  for slice_idx in range(num_slices):
    index = slice_idx*stride - 1 if slice_idx else slice_idx*stride
    image_slice = img_3d[0,index,:, :, :]
    slices.append(image_slice)

    filename = 'generated_img_{}.png'.format(slice_idx)
    png_file = os.path.join(dir.encode(), filename.encode())

    image_slice = np.ceil(image_slice * 255)

    png_data = tf.image.encode_png(image_slice)

    LOGGER.debug(png_file)

    _op = tf.write_file(
        filename= tf.constant(png_file.decode(), dtype= tf.string),
        contents= png_data)

    sess.run(_op)

  _plot_img_slices(slices)


def _plot_img_slices(slices):
  """
  In the interactive mode plot the slices.
  """
  if not OPTIONS.interactive:
    return

  fig, axs = plt.subplots(1, len(slices))
  fig.suptitle('Volume slices along the x-dim:')
  cmap = 'cool'
  plt_list = []

  for slice in slices:
    image_slice = np.squeeze(slice)
    plt_list.append(axs[i].imshow(image_slice, cmap= cmap))
    axs[i].label_outer()

  fig.colorbar(slices[0], ax= axs, orientation= 'horizontal', fraction= .1)

  plt.show()


def _archive_model_and_test_output(dir):
  """
  This function saves the exported model in 'WORK_DIR' as an archive in the 'export' dir for permanent storage.
  This function overwrites existing archives.
  """
  filename = '{0}_{1}_v_{2}'.format(OPTIONS.archive_name, OPTIONS.export_tag, OPTIONS.model_version)

  # [TODO] instead of getcwdb() get root_dir from the 'APP_ROOT_DIR' variable.
  zip_file = os.path.join(os.getcwdb(), b'exports', filename.encode())
  shutil.make_archive(zip_file.decode(), 'zip', dir)


def prepare_cmd_args(parser = None):
  if parser is None:
    parser = argparse.ArgumentParser(prog= sys.argv[0], description= 'Test model import.')

  parser.add_argument('--n_latent_dim', type=int, default=16, help='dimension of the latent representation.')
  parser.add_argument('--img_shape', type=str, default="64,64,64,1", help='shape of the 3d-image.')
  parser.add_argument('--import_dir', type=str, default="", help='model folder', required=True)
  parser.add_argument('--model_version', type=str, default="1", help='model version. also the subfolder name under import_dir')
  parser.add_argument('--export_tag', type=str, default="serving", help='model version. also the subfolder name under import_dir')
  parser.add_argument('--archive_name', type=str, default="debug_model_lite", help='archive name of the eported model')
  parser.add_argument('--save_archive', type=bool, default=False, help='save export folder as a archive')
  parser.add_argument('--interactive', type=bool, default=False, help='interactive session')

  return parser.parse_known_args()


if __name__ == "__main__":
  try:
    # preapare the options
    OPTIONS, unknown_options = prepare_cmd_args()
    LOGGER.message('...PARSED KNOWN ARGUMENTS...')
    LOGGER.message(OPTIONS)

    # check dir exitence
    if not os.path.exists(OPTIONS.import_dir):
      LOGGER.error('Model directory does not exist. Specify proper dir.\n\
          [--import_dir]={}'.format(OPTIONS.import_dir))
      sys.exit(-1)

    eval_model_ver_dir =  os.path.join(OPTIONS.import_dir, OPTIONS.model_version)
    if not os.path.exists(eval_model_ver_dir):
      LOGGER.error('Model version does not exist. \
          Specify proper version.\n\
          [--model_version]={0}\n\
          [eval_model_ver_dir]={1}'.format(
              OPTIONS.model_version,
              eval_model_ver_dir))
      sys.exit(-1)

    if not tf.saved_model.loader.maybe_saved_model_directory(eval_model_ver_dir):
      LOGGER.error('No exported model found in the provided path. \n\
          [eval_model_ver_dir]={}'.format(eval_model_ver_dir))

    # import >> test >> archive
    OPTIONS.img_shape = [ int(dim) for dim in OPTIONS.img_shape.split(',')]
    import_test_archive_model(eval_model_ver_dir)

  except:
    LOGGER.error(".... ERROR DURING MODEL IMPORT TEST ....")
    LOGGER.error(sys.exc_info()[0])
    raise
