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
     --n_gpu=2 \
     --param_device=cpu \
     --worker_device=gpu \
     --work_dir=$PWD \
     --batch_size=1000 \
     --n_epochs=30 \
     --data_file=__data_file_prefix__ \
     --data_dir=/root/data_dir \
     --model_tag=aaegan \
     --model_version=v_00_01 \
     --data_tag=hdf5 \
     #--fresh_training=True \
     #--check_only_graph_creation=True