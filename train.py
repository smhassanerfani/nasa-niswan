import os
import time
import json
import random
import argparse
import inspect
import numpy as np
import warnings
from sklearn.metrics import r2_score

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import E33OMA, E33OMA90D
from model import Discriminator, Generator, initialize_weights
from utils import seed, load_checkpoint, save_checkpoint, val_loop, LoggerDecorator


def main(args):
    
    warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar")

    since = time.time()
    
    print(f"{args.model} is deployed on {torch.cuda.get_device_name(0)}")
    
    # Loading model
    generator = Generator(in_channels=args.in_channels, features=64).cuda()
    
    # Initializing the model weights
    initialize_weights(generator)

    # Dataloader
    if args.dataset == 'E33OMA':
        train_dataset = E33OMA(period='train', species=args.species, padding=args.input_size, in_channels=args.in_channels, transform=args.transform)
        val_dataset   = E33OMA(period='val',   species=args.species, padding=args.input_size, in_channels=args.in_channels, transform=args.transform)
    
    if args.dataset == 'E33OMA90D':
        train_dataset = E33OMA90D(period='train', species=args.species, padding=args.input_size, in_channels=args.in_channels, transform=args.transform)
        val_dataset   = E33OMA90D(period='val',   species=args.species, padding=args.input_size, in_channels=args.in_channels, transform=args.transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Initializing the loss function and optimizer
    optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=tuple(args.betas))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.5)

    loss_func1 = nn.MSELoss() # reduction='none'
    loss_func2 = nn.L1Loss() # reduction='none'
    
    if args.use_checkpoint:
        load_checkpoint(f'{args.restore_from}/generator.pth.tar', generator, optimizer, args.learning_rate)
    
    logger = {'MSELoss': list(), 'r2_score': list(), 'r2_score_val': list()}
    
    for epoch in range(1, args.num_epochs + 1):
       
        generator.train()

        batch_loss = 0.0
        batch_r2   = 0.0
        
        for batch, (X, y) in enumerate(train_dataloader, 1):
            
            # GPU deployment
            X = X.cuda()
            y = y.cuda()

            # Training Generator            
            pred = generator(X)
            pred = pred[:, :, 83:83+90, 56:56+144]
            
            # Compute Loss Function
            loss = loss_func1(y, pred) + loss_func2(y, pred)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            batch_loss += loss.item()
            batch_r2   += r2_score(y.detach().cpu().numpy().flatten(), pred.detach().cpu().numpy().flatten())

        logger['MSELoss'].append(batch_loss / len(train_dataloader))
        logger['r2_score'].append(batch_r2  / len(train_dataloader))
        
        
        scheduler.step()
        
        logger['r2_score_val'].append(val_loop(val_dataloader, generator))
        
        print(f"Epoch: {epoch}, Loss: {logger['MSELoss'][-1]:.5f}, R2T: {logger['r2_score'][-1]:.5f}, R2V: {logger['r2_score_val'][-1]:.5f}")
        
        if epoch % 10 == 0:
            
            current_epoch_directory = os.path.join(args.snapshot_dir, f'epoch-{epoch:003d}')

            try:
                os.makedirs(current_epoch_directory)
            except FileExistsError:
                pass

            print('Learning Rate:', scheduler.get_last_lr())
            save_checkpoint(generator, optimizer,  os.path.join(current_epoch_directory, 'generator.pth.tar'), scheduler.get_last_lr(), epoch)
    
    with open(os.path.join(args.snapshot_dir, "logger.npy"), mode = 'wb') as f:
        np.save(f, np.array(logger['MSELoss']))
        np.save(f, np.array(logger['r2_score']))
        
        np.save(f, np.array(logger['r2_score_val']))
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


def get_arguments(
    MODEL='E33OMA-00',
    SPECIES='bcb',
    LEARNING_RATE=1.0E-04,
    DATASET='E33OMA',
    IN_CHANNELS=5,
    TRANSFORM=False,
    NUM_EPOCHS=50,
    INPUT_SIZE=256,
    BATCH_SIZE=4,
    NUM_WORKERS=1,
    SCHEDULER_STEP=10,
    BETAS=(0.5, 0.999),
    USE_CHECKPOINT=False
):
    SNAPSHOT_DIR = os.path.join('/home/serfani/serfani_data0/snapshots', MODEL)
    RESTORE_FROM = os.path.join('/home/serfani/serfani_data0/snapshots', MODEL)

    parser = argparse.ArgumentParser(description=f"Training {MODEL} on E33OMA.")
    parser.add_argument("--model", type=str, default=MODEL,
                        help=f"Model Name: {MODEL}")
    parser.add_argument("--species", type=str, default=SPECIES,
                        help=f"Name of aerosol species.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help=f"The name of dataset.")
    parser.add_argument("--in-channels", type=int, default=IN_CHANNELS,
                        help="Number of input channels of the model.")
    parser.add_argument("--transform", action="store_true", default=TRANSFORM,
                        help="Whether to transform data for training.")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="Number of epochs for training.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of s")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of workers for multithreading dataloader.")
    parser.add_argument("--scheduler-step", type=int, default=SCHEDULER_STEP,
                        help="Scheduler step of the optimizer.")
    parser.add_argument("--betas", nargs=2, type=float, default=BETAS,
                        help="Exponential decay rates for the 1st and 2nd moments of the optimizer.")
    parser.add_argument("--use-checkpoint", action="store_true", default=USE_CHECKPOINT,
                        help="Whether to use checkpointing during training.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where to restore the model parameters.")
    
    try:
        os.makedirs(SNAPSHOT_DIR)

    except FileExistsError:
        pass

    print('Working Directory:', SNAPSHOT_DIR)

    # Get the names and default values of input arguments
    arguments = inspect.signature(get_arguments).parameters
    defaults = {arg_name: param.default for arg_name, param in arguments.items()}

    defaults['SNAPSHOT_DIR'] = SNAPSHOT_DIR
    defaults['RESTORE_FROM'] = RESTORE_FROM

    # Dump the default arguments into a JSON file
    with open(os.path.join(SNAPSHOT_DIR, 'configurations.json'), 'w') as outfile:
        json.dump(defaults, outfile, indent=4)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    
    wrapper_call = LoggerDecorator(os.path.join(args.snapshot_dir, 'logger.log'))
    wrapper = wrapper_call(main)
    wrapper(args)