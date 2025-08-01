import os
# Set environment variables to help manage CUDA memory
os.environ['CUDA_VISIBLE_DEVICES'] = "MIG-214cfb66-c8e5-57f2-b101-90f2cca83fad"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set the number of threads for OpenMP and MKL
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import yaml
import argparse
from datetime import datetime

from model import STMLightning
from dataset import E33OMAModule

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

pl.seed_everything(42)

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

def main(args):

    # Retrieve hyperparameters
    with open(args.configdir, 'r') as config_filepath:
        hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)
        
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']
    
    # Get the current date and time
    now = datetime.now()
    formatted_date_time = now.strftime("%m%d%Y-%H%M")

    model_args['log_name'] = model_args['model_name'] + '-' + formatted_date_time
    # model_args['log_name'] = 'ConvLSTM-07032025-1434'

    # checkpoint_path = None
    checkpoint_path = '/home/serfani/serfani_data1/snapshots/ConvLSTM-07032025-1434/version_0/checkpoints/epoch=49-step=393700.ckpt'

    # Initialize model
    # model = STMLightning(model_args=model_args, data_args=data_args)
    model = STMLightning.load_pretrained(
        checkpoint_path=checkpoint_path,
        model_args=model_args,
        data_args=data_args
    )
    dm = E33OMAModule(data_args)

    # Initialize logger with the consistent log_name
    tb_logger = TensorBoardLogger(save_dir=model_args['log_dir'], name=model_args['log_name'], version=0)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        # dirpath=os.path.join(model_args['log_dir'], model_args['log_name'], 'checkpoints'),
        save_top_k=1,
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=model_args['epochs'],
        accelerator="gpu",
        devices=1,
        precision=32,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True
        )

    trainer.fit(model, datamodule=dm, ckpt_path=None)
    trainer.validate(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configdir', help='Provide the filepath string to the model config...', default='configs/ConvLSTM-03102025.yaml')
    
    args = parser.parse_args()
    main(args)