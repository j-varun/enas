#!/bin/bash
export PYTHONPATH="$(pwd)"
# from 2018_09_15_0953_stacking_outputs_rotation_search_no_root_no_msle.txt
# Epoch 300
# [1 2 0 1 2 3 1 0 0 3 2 0]                                                                                                                                                   │·························
# [0 1 1 1 0 2 2 0 1 3 1 3]
# val_acc=0.9375                                                                                                                                                              │·························
# controller_loss=3453.15795898                                                                                                                                               │·························
# mse=0.00610682461411                                                                                                                                                        │·························
# angle_error=0.122636593878                                                                                                                                                  │·························
# mae=0.12954197824
# -------------------------
fixed_arc="1 2 0 1 2 3 1 0 0 3 2 0"
fixed_arc="$fixed_arc 0 1 1 1 0 2 2 0 1 3 1 3"

python enas/cifar10/main.py \
  --data_format="NHWC" \
  --search_for="micro" \
  --reset_output_dir \
  --output_dir="2018_09_17_1723_stacking_outputs_rotation_without_root_msle_final" \
  --batch_size=64 \
  --num_epochs=630 \
  --log_every=50 \
  --eval_every_epochs=10 \
  --child_fixed_arc="${fixed_arc}" \
  --child_use_aux_heads \
  --child_num_layers=10 \
  --child_out_filters=36 \
  --child_num_branches=5 \
  --child_num_cells=3 \
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
  --dataset="stacking" \
  --height_img 64 \
  --width_img 64 \
  --rotation_only \
  --max_loss=2 \
  --use_msle \
  --one_hot_encoding \
  "$@"