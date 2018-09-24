#
# ---------------------------------------------------------
# Copyright 2018-present (c) Automatos Studios. All Rights Reserved.
# ---------------------------------------------------------
# 
# @file       model_factory.py
# @author     cosmoplankton@automatos.studio
#

"""
Implementation of ModelFactory.
* Import the appropriate Model module as per the tag and version.
"""


import importlib

from utils import global_logging as logging
LOGGER = logging.get_logger()


class ModelFactory:
  """
  Import the appropriate Model module as per the tag and version.
  """
  def __init__(self, **kwargs):
    pass

  @staticmethod
  def get_model(dataservice, params, tag='aaegan', version='v_00_01'):
    """
    Return the Model object based on the requested version.
    """
    try:
      _model_module = importlib.import_module(
          ".model_" + tag + '_' + version,
          package='models')
    except ImportError:
      LOGGER.error("Import error during loading model. Check whether model tag or version is correct.")
      raise

    return _model_module.Model(dataservice= dataservice, params= params)

