#!/bin/bash

# Change to the directory where your Python script is located
cd /home/serfani/projects/nasa-niswan/

# Activate the virtual environment
source /home/serfani/projects/nasa-niswan/.venv/bin/activate

DATEDIR=$(date +"%Y%m%d-%H%M");
MODEL=LSTM-64K5.32K3.16K3-E33OMA90D-BCB-$DATEDIR; # [UNet/PIX2PIX/LSTM-<HIDDEN-STATES>K<KERNEL-SIZE>]-[E33OMA/E33OMA90D]-[5C/6C]-[BCB/SST/CLY]-$DATEDIR;

# Run the Python script with arguments
python train.py \
    --model $MODEL\
    --species bcb \
    --learning-rate 1.0E-03 \
    --dataset E33OMA90D \
    --in-channels 5 \
    --hidden-channels 64 32 16 \
    --kernel-size 5 3 3 \
    --num-layers 3 \
    --sequence-length 48 \
    --num-epochs 50 \
    --input-size 100 154 \
    --batch-size 8 \
    --num-workers 1 \
    --scheduler-step 25 \
    --betas 0.5 0.999 \
    --snapshot-dir /home/serfani/serfani_data0/snapshots/$MODEL \
    --restore-from /home/serfani/serfani_data0/snapshots/$MODEL \
    # --transform \
    # --use-checkpoint \

# Deactivate the virtual environment
deactivate
