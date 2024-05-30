#!/bin/bash

# Change to the directory where your Python script is located
cd /home/serfani/projects/nasa-niswan/

# Activate the virtual environment
source /home/serfani/projects/nasa-niswan/.venv/bin/activate

DATEDIR=$(date +"%Y%m%d-%H%M");
MODEL=LSTM-3C-E33OMA-CRNN-BCB-$DATEDIR; # [UNet/PIX2PIX/LSTM-<NUM-CELLS>C]-[E33OMA/E33OMA90D]-[5C/6C]-[BCB/SST/CLY]-$DATEDIR;

# Run the Python script with arguments
python train.py \
    --model $MODEL\
    --species bcb \
    --learning-rate 1.0E-03 \
    --dataset E33OMA-CRNN \
    --in-channels 5 \
    --hidden-channels 10 \
    --kernel-size 3 \
    --num-layers 3 \
    --sequence-length 48 \
    --num-epochs 50 \
    --input-size 256 \
    --batch-size 16 \
    --num-workers 1 \
    --scheduler-step 10 \
    --betas 0.5 0.999 \
    --snapshot-dir /home/serfani/serfani_data0/snapshots/$MODEL \
    --restore-from /home/serfani/serfani_data0/snapshots/$MODEL \
    # --transform \
    # --use-checkpoint \

# Deactivate the virtual environment
deactivate
