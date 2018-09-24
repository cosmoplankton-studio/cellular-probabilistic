#!/usr/bin/env bash

python ./trainers/run_trainer_async.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2224 \
     --job_name=ps \
     --task_idx=0 \
     --work_dir=$PWD \
     --data_file=_placeholder_