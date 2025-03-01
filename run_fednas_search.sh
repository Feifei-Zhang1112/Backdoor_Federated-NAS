#!/usr/bin/env bash

GPU=$1
MODEL=$2
# homo; hetero
DISTRIBUTION=$3
ROUND=$4
EPOCH=$5
BATCH_SIZE=$6

mpiexec -n 21 python3 ./main.py \
  --gpu $GPU \
  --model $MODEL \
  --dataset tiny \
  --partition $DISTRIBUTION  \
  --client_number 20 \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE\
  --attack_type dba \
  --number_of_adversaries 2\
  --defense_method None \
  --start_gpu 0 \
  --offline no \
  --init_channels 32 \
  --learning_rate 0.006 \
  --search_space "TINYGENO" \
  --portion 1 \
  --attack_scale 1 \
  --poison_type "MGDA" \
  --notes  "100 2  GENOSPACE" \
 # --path "/home/vipuser/"