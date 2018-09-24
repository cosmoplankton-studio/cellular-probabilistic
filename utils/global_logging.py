#
# ---------------------------------------------------------
# Copyright 2018-present (c) Automatos Studios. All Rights Reserved.
# ---------------------------------------------------------
# 
# @file     global_logging.py
# @author   cosmoplankton@automatos.studio
#

"""
* Implements the global logger.
* Internally uses tf.logging and file io operations.
"""

import tensorflow as tf


class LogMode:
  OFF = 'off';        # all off
  TF_LOGGING = 'tf';  # log using tf.logging
  FILE = 'log-file';  # log using tf.logging and write to log file.



class Log:
  """
  Provides the interface for the global logger.
  """
  def __init__(self, verbosity= tf.logging.DEBUG):
    self._data = list()
    self._mode = LogMode.TF_LOGGING
    tf.logging.set_verbosity(verbosity)

  def set_mode(self, mode):
    self._mode = mode

  def message(self, msg):
    msg = "[MESSAGE]  {}".format(msg)
    tf.logging.info(msg)
    self._flush(msg)

  def error(self, error):
    msg = "[ERROR]  {}".format(error)
    tf.logging.info(msg)
    self._flush(msg)

  def debug(self, msg):
    msg = "[DEBUG]  {}".format(msg)
    tf.logging.debug(msg)
    self._flush(msg)

  def _flush(self, msg):
    """
    Flush to log file or remote location.
    Flushes every 100 messages.
    """
    if self._mode == LogMode.FILE:
      raise NotImplementedError
      if len(self._data) >= 100:
        # write to file/remote
        self._data.clear()
      else:
        self._data.append(msg)

  def DEBUG_VARIABLES(self):
    # [DEBUG] // Track variables to make sure they are getting reused.
    self.debug('----------- GLOBAL_VARIABLES START -----------')

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    for var_data in zip(range(len(var_list)), var_list):
      self.debug('[{0}]__ {1}'.format(var_data[0], var_data[1]))

    self.debug('-----------  GLOBAL_VARIABLES END  -----------')

  def DEBUG_LOSSES(self):
    # [DEBUG] // Track losses.
    self.debug('----------- LOSSES START -----------')

    losses = tf.get_collection(tf.GraphKeys.LOSSES)

    for loss_data in zip(range(len(losses)), losses):
      self.debug('[{0}]__ {1}'.format(loss_data[0], loss_data[1]))

    self.debug('-----------  LOSSES END  -----------')



# just use a global logger.
# [TODO]: common logger for distributed learning.

# g_logger = Log(verbosity= tf.logging.DEBUG)
g_logger = Log(verbosity= tf.logging.INFO)


def get_logger():
  return g_logger