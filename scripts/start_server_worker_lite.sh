#!/usr/bin/env bash

###########################
## ADD --check_only_graph_creation FOR ONLY GRAPH CREATION TEST
## ADD --fresh_training FOR DELETING TMP_WORK_DIR
###########################

python ./trainers/run_trainer_async.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2224 \
     --job_name=worker \
     --task_idx=0 \
     --n_gpu=1 \
     --param_device=cpu \
     --worker_device=cpu \
     --work_dir=$PWD \
     --batch_size=2 \
     --n_epochs=3 \
     --data_file=point_cloud_64px_dataitems_20_uint8.tfrecord \
     --data_dir=$PWD/debug_data \
     --model_tag=aaegan \
     --model_version=v_lite \
     --data_tag=debug_data \
     --export_version=1 \
     --fresh_training=True \
     #--check_only_graph_creation=True