#!/bin/bash

# Change to the directory where your Python script is located
cd /home/serfani/projects/nasa-niswan/

# Activate the virtual environment
source /home/serfani/projects/nasa-niswan/.venv/bin/activate

DATEDIR=$(date +"%Y%m%d-%H%M");
MODEL=PIX2PIX-E33OMA90D-BCB-$DATEDIR; # [UNet/PIX2PIX]-[E33OMA/E33OMA90D]-[BCB/SST/CLY]-$DATEDIR;

# Run the Python script with arguments
python train.py \
    --model $MODEL\
    --species bcb \
    --learning-rate 1.0E-04 \
    --dataset E33OMA90D \
    --in-channels 5 \
    --num-epochs 50 \
    --input-size 256 \
    --batch-size 4 \
    --num-workers 1 \
    --scheduler-step 10 \
    --betas 0.5 0.999 \
    --snapshot-dir /home/serfani/serfani_data0/snapshots/$MODEL \
    --restore-from /home/serfani/serfani_data0/snapshots/$MODEL \
    # --transform \
    # --use-checkpoint \

# Deactivate the virtual environment
deactivate
