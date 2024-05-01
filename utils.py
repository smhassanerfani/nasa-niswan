
import time
import logging
import datetime
import random
import numpy as np
import torch
from functools import wraps
from sklearn.metrics import r2_score

def save_checkpoint(model, optimizer, filename, learning_rate=None, epoch=None):

    print('Saving Checkpoint...')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': learning_rate,
        'epoch': epoch
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None):
    
    checkpoint = torch.load(checkpoint_file)
    print('Number of Epochs: ', checkpoint['epoch'])
    print('Learning Rate: ', checkpoint['learning_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      
        if lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        elif checkpoint['learning_rate'] is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = checkpoint['learning_rate']

def val_loop(dataloader, model):

    model.eval()
    r2 = 0.0
    
    with torch.no_grad():
        for counter, (X, y) in enumerate(dataloader, 1):

            # GPU deployment
            X = X.cuda()
            y = y.cuda()

            # Compute prediction and loss
            pred = model(X)
            pred = pred[:, :, :90, :144]
            
            r2 += r2_score(y.detach().cpu().numpy().flatten(), pred.detach().cpu().numpy().flatten())

    return r2 / len(dataloader)


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LoggerDecorator(object):
    def __init__(self, log_file, level=logging.INFO):
        self.log_file = log_file
        self.level = level

    def __call__(self, func):
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            
            logger = logging.getLogger()
            logger.setLevel(self.level)
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            tic = time.time()
            result = func(*args, **kwargs)
            toc = time.time() - tic

            # Convert time difference to conventional format
            compiling_time = str(datetime.timedelta(seconds=toc))
            logger.info(f"Run configuration: {args, kwargs}, Compiling time: {compiling_time}")

            logger.removeHandler(handler)
            
            return result
        
        return wrapper