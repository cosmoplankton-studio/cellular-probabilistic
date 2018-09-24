#
# ---------------------------------------------------------
# Copyright 2018-present (c) Automatos Studios. All Rights Reserved.
# ---------------------------------------------------------
# 
# @file       model_aaegan_v_lite.py
# @author     cosmoplankton@automatos.studio
#

"""
#===================================================
# This implementation is used for debugging.
#===================================================
"""

from networks import network_aae_v_lite as networks
from models import model_aaegan_base as base

from utils import global_logging as logging
LOGGER = logging.get_logger()



class Model(base.Model):
  """
  Derived class used for debugging.
  """

  def __init__(self, dataservice, params):
    super().__init__(dataservice, params)


  def setup(self):
    """
    setup the model component networks.
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
