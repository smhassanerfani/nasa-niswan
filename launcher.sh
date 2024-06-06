#!/bin/bash

# Change to the directory where your Python script is located
cd /home/serfani/projects/nasa-niswan/

# Activate the virtual environment
source /home/serfani/projects/nasa-niswan/.venv/bin/activate

DATEDIR=$(date +"%Y%m%d-%H%M");
MODEL=LSTM-3C16-E33OMA90D-CRNN-BCB-$DATEDIR; # [UNet/PIX2PIX/LSTM-<NUM-CELLS>C-<NUM-STATES>]-[E33OMA/E33OMA90D]-[5C/6C]-[BCB/SST/CLY]-$DATEDIR;

# Run the Python script with arguments
python train.py \
    --model $MODEL\
    --species bcb \
    --learning-rate 1.0E-03 \
    --dataset E33OMA90D-CRNN \
    --in-channels 5 \
    --hidden-channels 16 \
    --kernel-size 3 \
    --num-layers 3 \
    --sequence-length 48 \
    --num-epochs 100 \
    --input-size 100 154 \
    --batch-size 16 \
    --num-workers 1 \
    --scheduler-step 20 \
    --betas 0.5 0.999 \
    --snapshot-dir /home/serfani/serfani_data0/snapshots/$MODEL \
    --restore-from /home/serfani/serfani_data0/snapshots/$MODEL \
    # --transform \
    # --use-checkpoint \

# Deactivate the virtual environment
deactivate
