#
# ---------------------------------------------------------
# Copyright 2018-present (c) Automatos Studios. All Rights Reserved.
# ---------------------------------------------------------
# 
# @file       run_trainer_async.py
# @author     cosmoplankton@automatos.studio
#

"""
@description:
  Implementation of a distributed trainer with below properties:
    (a) single-host-multi-device. (each replica has this assumed configuration)
    (b) synchronous gradient averaging for devices.
    (c) asynchronous training for replicas.
    (d) Between-graph replication.
    LIMITATIONS:
    (a) Synchronous replica update not fully implemented.
    (b) parameter servers use 'cpu', 'gpu' as parameter server not supported.
    (c) Multiple 'cpu' on same worker not supported.
"""


# import dependencies

import os
import sys
import pickle
import json
import shutil
import importlib
import math
import pdb

import argparse
import time
import datetime

import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())


# import the internal modules
from dataservices import data_provider as dp
from models import model_factory as mf
from models import model_defines as md

from utils import global_logging as logging
LOGGER = logging.get_logger()

OPTIONS = None
TIME_STAMP = None


def prepare_cmd_args(parser= None):
  """
  parse the command line arguments
  """
  if parser is None:
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='Train the models over a distributed cluster.')

  #parser.add_argument('--chief_host', type=str, default="localhost:2222", help='chief worker server, initialization + worker')
  parser.add_argument('--ps_hosts', type=str, default="localhost:2223", help='comma separated string of ps servers')
  parser.add_argument('--worker_hosts', type=str, default="localhost:2224,localhost:2225", help='comma separated string of worker servers')
  parser.add_argument('--job_name', type=str, default="ps", help='job name for this server ("chief"/"worker"/"ps")')
  parser.add_argument('--task_idx', type=int, default=0, help='task index for a local tf server')
  parser.add_argument('--n_gpu', type=int, default=1, help='number of gpus to use.')
  parser.add_argument('--sync_replicas', type=bool, default=False, help='synchronous distributed training.')
  parser.add_argument('--param_device', type=str, default='cpu', help='parameter device.')
  parser.add_argument('--worker_device', type=str, default='gpu', help='worker device.')
  parser.add_argument('--check_only_graph_creation', type=bool, default=False, help='only check graph creation, do not start training')

  parser.add_argument('--n_latent_dim', type=int, default=16, help='number of latent dimensions')
  parser.add_argument('--n_classes', type=int, default=0, help='number of classes')
  # [source for defaults] - http://dx.doi.org/10.1101/238378
  parser.add_argument('--lrate_encoder', type=float, default=0.0002, help='learning rate for the encoder')
  parser.add_argument('--lrate_decoder', type=float, default=0.0002, help='learning rate for the decoder')
  parser.add_argument('--lrate_encoder_d', type=float, default=0.01, help='learning rate for the encoder discriminator')
  parser.add_argument('--lrate_decoder_d', type=float, default=0.0002, help='learning rate for the decoder discriminator')
  parser.add_argument('--lambda_encoder_d', type=float, default=1E-4, help='scalar applied to the update gradient from encoder discriminator')
  parser.add_argument('--lambda_decoder_d', type=float, default=1E-5, help='scalar applied to the update gradient from decoder discriminator')

  parser.add_argument('--fresh_training', type=bool, default=False, help='delete existing TMP_WORK_DIR and start fresh.')
  parser.add_argument('--work_dir', type=str, default=None, help='parent working directory', required=True)
  parser.add_argument('--data_dir', type=str, default="data_debug", help='data dir path. this could be a local dir or s3-url')
  parser.add_argument('--data_file', type=str, default=None, help='data file name. for single file this is the filename. for multiple files this is the prefix.')
  parser.add_argument('--data_src', type=str, default="local", help='data_dir points to local/s3-bucket. [local, s3]')
  parser.add_argument('--export_version', type=str, default="1", help='version of exported model')
  parser.add_argument('--export_dir', type=str, default="export.tf", help='name of export dir')
  # parser.add_argument('--remote_dir', type=str, default="", help='remote url to s3-bucket where logs, exports, etc. will be saved.')

  parser.add_argument('--run_mode', type=str, default='TRAIN', help='train, or test, or evaluate')
  parser.add_argument('--model_tag', type=str, default='aaegan', help='string tag of the model to be trained')
  parser.add_argument('--model_version', type=str, default='v_00_01', help='version of the model to be trained')
  parser.add_argument('--data_tag', type=str, default='debug_data', help='datapipeline to use to fetch')
  parser.add_argument('--n_epochs', type=int, default=20, help='total number of epochs')
  parser.add_argument('--batch_size', type=int, default=1000, help='batch size per iteration of training. This will get re-distributed across multiple GPUs' )

  parser.add_argument('--export_as_texts', type=bool, default=True, help='export the models as text files' )

  return parser.parse_known_args()


def prepare_dir_and_files():
  """
  Prepare the directories and the files for training and exporting.
  """
  if OPTIONS.work_dir is None:
    raise ValueError('--work_dir --working directory is an mandatory argument.')

  work_dir_this_run = os.path.join(OPTIONS.work_dir, 'TF_WORK_DIR')
  OPTIONS.export_dir = os.path.join(work_dir_this_run, 'export.tf')
  OPTIONS.checkpoint_dir = os.path.join(work_dir_this_run, 'checkpoints.tf', OPTIONS.run_mode + '_' + TIME_STAMP)
  # OPTIONS.data_dir = os.path.join(OPTIONS.work_dir, OPTIONS.data_dir)

  OPTIONS.work_dir = work_dir_this_run

  if not os.path.exists(work_dir_this_run):
    os.mkdir(work_dir_this_run)
  elif OPTIONS.fresh_training:
    shutil.rmtree(path=work_dir_this_run, ignore_errors= True)
    os.mkdir(work_dir_this_run)

  if not os.path.exists(OPTIONS.export_dir):
    os.mkdir(OPTIONS.export_dir)

  OPTIONS.export_dir = os.path.join(OPTIONS.export_dir, OPTIONS.export_version)

  if os.path.exists(OPTIONS.export_dir):
    LOGGER.error('Model Export Version Exists. Specify a different version or clean existing version.')
    raise AssertionError('Model Export Version Exists. Specify a different version or clean existing version.')

  if not os.path.exists(OPTIONS.checkpoint_dir):
    os.makedirs(OPTIONS.checkpoint_dir)

  run_options_dump_file = '{0}/options.pkl'.format(work_dir_this_run)

  # always overwrite the options dump file after keeping a back-up
  if os.path.exists(run_options_dump_file):
    shutil.copyfile(run_options_dump_file, '{0}_{1}'.format(run_options_dump_file, TIME_STAMP))

  handle = open(run_options_dump_file, 'wb')
  pickle.dump(OPTIONS, handle)
  handle.close()


def prepare_data_params():
  """
  Setup dir and files related to the dataservice.
  """
  if OPTIONS.data_file is None:
    LOGGER.error('--data_file name is an mandatory argument.')
    exit(-1)

  if OPTIONS.data_src == "local" and not os.path.exists(OPTIONS.data_dir):
    LOGGER.error('data_dir does not exists')
    exit(-1)


def create_single_host_cluster():
  return tf.train.Server.create_local_server()


def create_multi_host_cluster():
  # [TODO] - Kubernete cluster.
  # [source] - https://github.com/tensorflow/ecosystem
  raise NotImplementedError


def main(argv=None):
  """
  * Function for the training loop and the export.
  * This function gets called by both 'ps' and 'worker' servers.
  * [source-tf-distributed] https://www.tensorflow.org/deploy/distributed
  """

  # chief_spec = OPTIONS.chief_host.split(',')
  ps_spec = OPTIONS.ps_hosts.split(',')
  worker_spec = OPTIONS.worker_hosts.split(',')

  cluster_def = {
    "ps": ps_spec,
    "worker": worker_spec
  }

  cluster_spec = tf.train.ClusterSpec(cluster_def)
  server = tf.train.Server(cluster_spec, job_name= OPTIONS.job_name, task_index= OPTIONS.task_idx, start= False)

  OPTIONS.is_chief = False
  # OPTIONS.n_replica_workers = len(worker_spec) + len(chief_spec)
  OPTIONS.n_replica_workers = len(worker_spec)


  if OPTIONS.job_name == "ps":
    server.start()
    server.join()
    return
  elif OPTIONS.job_name == "chief":
    OPTIONS.is_chief = True
    main_worker(server= server, cluster= cluster_spec, argv= argv)
  elif OPTIONS.job_name == "worker":
    server.start()
    if(OPTIONS.task_idx == 0):
      OPTIONS.is_chief = True
    main_worker(server= server, cluster= cluster_spec, argv= argv)



def main_worker(server, cluster, argv=None):
  """
  main() function invoked for training by the workers.
  """
  # prepare the data directory
  prepare_data_params()

  # prepare relevant directories and files
  prepare_dir_and_files()

  # prepare graph and pin variables/operations to appropriate devices
  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % OPTIONS.task_idx,
    cluster=cluster)):

    LOGGER.message('----- GRAPH CREATION STARTED WITHOUT ERROR -----')

    # dataservice initialization
    data_provider = dp.DataProvider()
    dataservice_inst = data_provider.get_dataservice()
    OPTIONS.img_pipeline = dataservice_inst.create_data_pipeline(tag= OPTIONS.data_tag, params= OPTIONS)
    n_max_dataitems = dataservice_inst.get_data_pipeline(OPTIONS.img_pipeline).get_max_dataitems()

    OPTIONS.n_iters_per_epoch = math.floor(n_max_dataitems / OPTIONS.batch_size)
    OPTIONS.n_max_iters = OPTIONS.n_epochs * OPTIONS.n_iters_per_epoch
    LOGGER.message('----- n_max_iters: {} -----'.format(OPTIONS.n_max_iters))

    dataservice_inst.setup(params= OPTIONS)

    # model initialization
    model_provider = mf.ModelFactory()
    model_instance = model_provider.get_model(
        tag= OPTIONS.model_tag,
        version= OPTIONS.model_version,
        dataservice= dataservice_inst,
        params= OPTIONS)

    error = model_instance.setup()
    model_train_op, optim_global_steps = model_instance.build_model()

    hooks_train = model_instance.get_hooks(md.Hooks.TRAIN)
    hooks_summary = model_instance.get_hooks(md.Hooks.SUMMARY)
    hooks_checkpoint = model_instance.get_hooks(md.Hooks.CHECKPOINT)
    hooks_logging = model_instance.get_hooks(md.Hooks.LOG)

    # create the global step
    global_step = tf.train.get_or_create_global_step()
    with tf.control_dependencies([model_train_op]):
      update_g_step = global_step.assign_add(1)

  LOGGER.message('----- GRAPH CREATION COMPLETE WITHOUT ERROR -----')

  if OPTIONS.check_only_graph_creation:
    return

  # logging tensors
  run_outputs_info = {
      'model_train_op': model_train_op,
      'global_step': update_g_step,
      'optim_enc_d_step': optim_global_steps[0],
      'optim_dec_d_step': optim_global_steps[1],
      'optim_enc_step': optim_global_steps[2],
      'optim_dec_step': optim_global_steps[3],
  }
  step_logging = tf.train.LoggingTensorHook(tensors=run_outputs_info, every_n_iter=1)
  hooks_logging.append(step_logging)

  # prepare training hooks
  hooks = list()
  hooks.append(tf.train.StopAtStepHook(last_step=OPTIONS.n_max_iters))
  hooks.extend(hooks_train)
  hooks.extend(hooks_summary)
  hooks.extend(hooks_logging)

  # compute resources
  compute_config = tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:%d' % OPTIONS.task_idx])
  compute_config.gpu_options.per_process_gpu_memory_fraction = 0.3
  compute_config.allow_soft_placement=True
  # compute_config.gpu_options.allow_growth=True

  # Train the model
  LOGGER.message('----- IS CHIEF: {} -----'.format(OPTIONS.is_chief))

  with tf.train.MonitoredTrainingSession(
      master= server.target,
      is_chief= OPTIONS.is_chief,
      hooks= hooks,
      config= compute_config,
      checkpoint_dir=OPTIONS.checkpoint_dir,
      save_checkpoint_steps= 1,
      save_summaries_steps= 1) as mon_sess:

    while not mon_sess.should_stop():
      _ = mon_sess.run(run_outputs_info)

    # Export after training
    # NOTE: using '_unsafe_unfinalize()' && '_tf_sess()', need to figure out safe export for monitored session.
    if OPTIONS.is_chief:
      g = mon_sess.graph
      g._unsafe_unfinalize()
      model_instance.export_serving(mon_sess._tf_sess(), OPTIONS.export_dir, as_text= OPTIONS.export_as_texts)
      g.finalize()


def log_init_messages(session):
  # log devices
  logger.debug('----- devices available in the session -----')
  devices = session.list_devices()
  for d in devices:
    logger.message(d.name)

  # log unintialized variables
  # session.run(tf.global_variables_initializer())
  LOGGER.debug('----- UNINITIALIZED TRAINABLE VARIABLES -----')
  LOGGER.error(session.run(tf.report_uninitialized_variables()))



if __name__ == "__main__":
  try:
    # prepare timestamp
    TIME_STAMP = datetime.datetime.now().strftime("%Y-%m-%d-%HHrs-%MMins-%SSecs")

    # preapare the options
    OPTIONS, unknown_options = prepare_cmd_args()
    LOGGER.message('----- PARSED KNOWN ARGUMENTS START -----')
    LOGGER.message(OPTIONS)
    LOGGER.message('-----  PARSED KNOWN ARGUMENTS END  -----')

    tf.app.run(main= main, argv= [sys.argv[0]] + unknown_options)

  except SystemExit:
    LOGGER.message("----- APPLICATION DONE -----")
  except:
    LOGGER.error("----- ERROR TERMINATION -----")
    LOGGER.error(sys.exc_info()[0])
    raise


