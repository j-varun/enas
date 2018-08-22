#!/bin/bash

export PYTHONPATH="$(pwd)"


fixed_arc="0 1 1 0 1 4 0 0"
fixed_arc="$fixed_arc 0 1 1 2 0 0 0 2"

python enas/cifar10/main.py \
  --data_format="NHWC" \
  --search_for="micro" \
  --reset_output_dir \
  --data_path="data/cifar10" \
  --output_dir="stacking_outputs_rotation_final" \
  --batch_size=32 \
  --num_epochs=630 \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_fixed_arc="${fixed_arc}" \
  --child_use_aux_heads \
  --child_num_layers=10 \
  --child_out_filters=36 \
  --child_num_branches=5 \
  --child_num_cells=5 \
  --child_keep_prob=0.80 \
  --child_drop_path_keep_prob=0.60 \
  --child_l2_reg=2e-4 \
  --child_lr_cosine \
  --child_lr_max=1.0 \
  --child_lr_min=0.0001 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --nocontroller_training \
  --controller_search_whole_channels \
  --controller_entropy_weight=0.0001 \
  --controller_train_every=1 \
  --controller_sync_replicas \
  --controller_num_aggregate=10 \
  --controller_train_steps=50 \
  --controller_lr=0.001 \
  --controller_tanh_constant=1.50 \
  --controller_op_tanh_reduce=2.5 \
  --data_path="~/.keras/datasets/costar_block_stacking_dataset_v0.2/*success.h5f" \
  --dataset="stacking" \
  --height_img 96 \
  --width_img 96 \
  --rotation_only \
  --max_loss=2 \
  "$@"

