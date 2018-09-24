#!/usr/bin/env bash

python ./dataservices/data_debug_tfrecords.py \
     --data_dir=$PWD/debug_data \
     --data_file=point_cloud_64px_dataitems_20_uint8.tfrecord \
     --n_dataitems=20 \
     --img_dim=64 \
     --shape_dia=48 \
     --n_channels=1