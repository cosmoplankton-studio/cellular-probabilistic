#
# ---------------------------------------------------------
# Copyright 2018-present (c) Automatos Studios. All Rights Reserved.
# ---------------------------------------------------------
# 
# @file       data_provider.py
# @author     cosmoplankton@automatos.studio
#

"""
DataProvider Implementation:
  Imports the appropriate dataservice based on 'tag' and 'version'.
"""

import importlib

from utils import global_logging as logging
LOGGER = logging.get_logger()



class DataProvider:
  """
  Imports the appropriate dataservice based on 'tag' and 'version'.
  """
  def __init__(self, **kwargs):
    pass

  @staticmethod
  def get_dataservice(tag= None, version= None):
    """
    Return the Dataservice object based on the requested version.
    """
    if tag is None:
      tag = 'base'

    tag = tag.lower()
    try:
      if version is None:
        _ds_module = importlib.import_module(
            ".data_service_" + tag,
            package= 'dataservices')
      else:
        _ds_module = importlib.import_module(
            ".data_service_" + tag + '_' + version,
            package= 'dataservices')
    except ImportError:
      LOGGER.error('Dataservice import failed')
      raise

    return _ds_module.ServiceBase()
