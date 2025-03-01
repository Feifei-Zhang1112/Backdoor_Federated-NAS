#!/usr/bin/env bash

GPU=$1
# new added
STAGE=$2
DEFENSE_METHOD=$3
# homo; hetero
DISTRIBUTION=$4
ROUND=$5
EPOCH=$6
BATCH_SIZE=$7


mpiexec -n 21 python3 ./main.py \
  --gpu $GPU \
  --stage $STAGE \
  --model darts \
  --dataset cifar10 \
  --partition $DISTRIBUTION  \
  --client_number 20 \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --attack_type cerp \
  --number_of_adversaries 0\
  --defense_method ESFL \
  --start_gpu 0 \
  --offline no \
  --learning_rate 0.01 \
  --model_arch "CIFARSEARCH" \
  --init_channels 32 \
  --attack_scale  3\
  --poison_type "DATA" \
  --portion 0.1 \
  --notes "872 a3fl high " \
  --tags "changed_lr" \
  # --path "/home/vipuser/"
#!/usr/bin/env bash

# constrain, dba, a3fl, f3ba, cerp
# FEDRAD, CRFL, ESFL, SSD


