#!/usr/bin/env bash

python ./exports/import_test_archive.py \
     --import_dir=$PWD/TF_WORK_DIR/export.tf \
     --model_version=1 \
     --export_tag=serving \
     --save_archive=True \
     --n_latent_dim=16 \
     --img_shape=64,64,64,1