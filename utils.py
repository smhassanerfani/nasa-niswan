
import os
import time
import logging
import datetime
import random
import numpy as np
import torch
from functools import wraps
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import xarray as xr

# Plot Settings
plt.rcParams.update({
    # 'font.sans-serif': 'Comic Sans MS',
    'font.family': 'serif',
    'font.size': 12,
    'text.usetex': False   
})

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

def val_loop(args, dataloader, model):

    model.eval()
    r2 = 0.0
    
    with torch.no_grad():
        for counter, (X, y) in enumerate(dataloader, 1):

            # GPU deployment
            X = X.cuda()
            y = y.cuda()

            # Compute prediction and loss
            pred = model(X)

            if args.model.split('-')[0] in ['PIX2PIX', 'UNet']:
                pred = pred[:, :, 83:83+90, 56:56+144]
            
            elif args.model.split('-')[0] in ['LSTM']:
                pred = pred[0].squeeze()
            
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

def make_saving_path(root, format, name):
    return os.path.join(root, f'{name}.{format}')

def qqplot(ds_list, var, yax1='' ,axis_names=None, quantiles=None, save_figure=False, fformat=None, saving_path=None):

    ds = xr.open_mfdataset(ds_list)

    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"

    if ds[var].ndim == 3:
        ds = ds.weighted(weights).mean(dim=("lat", "lon"))
    
    if ds[var].ndim == 4:    
        ds = ds.isel(level=0).weighted(weights).mean(dim=("lat", "lon"))
        
    y_test = ds[var].isel(time=slice(0, 17520)).values
    y_pred = ds[var].isel(time=slice(17520, None)).values

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 3), constrained_layout=True)

    if axis_names is None:
        y_test_name='GT'
        y_pred_name='MODEL'
    else:
        y_test_name=axis_names[0]
        y_pred_name=axis_names[1]

    ax1.boxplot([y_test, y_pred])

    ax1.set_xticklabels([y_test_name, y_pred_name])
    ax1.tick_params(axis='x', labelrotation=0, labelsize=12)
    ax1.set_ylabel(yax1)
    ax1.grid(True)

    x1 = np.sort(y_test)
    y1 = np.arange(1, len(y_test) + 1) / len(y_test)
    ax2.plot(x1, y1, linestyle='-', alpha=0.8, label=y_test_name)

    x2 = np.sort(y_pred)
    y2 = np.arange(1, len(y_pred) + 1) / len(y_pred)
    ax2.plot(x2, y2, linestyle='-.', alpha=1, label=y_pred_name)
    ax2.legend()

    if quantiles is None:
        quantiles = min(len(y_test), len(y_pred))
    quantiles = np.linspace(start=0, stop=1, num=int(quantiles))

    x_quantiles = np.quantile(y_test, quantiles, method='nearest')
    y_quantiles = np.quantile(y_pred, quantiles, method='nearest')

    ax3.scatter(x_quantiles, y_quantiles)

    max_value = np.array((x_quantiles, y_quantiles)).max()
    min_value = np.array((x_quantiles, y_quantiles)).min()
    ax3.plot([min_value, max_value], [min_value, max_value], '--', color='black', linewidth=1.5)

    ax3.set_xlabel(y_test_name)
    ax3.set_ylabel(y_pred_name)

    if save_figure:
        plt.savefig(saving_path, format=f'{fformat}', bbox_inches='tight', pad_inches=0.1)
    
    plt.show()


def plot_global_ave(ds_list, var, save_figure=False, fformat=None, saving_path=None):
    ds = xr.open_mfdataset(ds_list)

    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"

    if ds[var].ndim == 3:
        ds = ds.weighted(weights).mean(dim=("lat", "lon"))
    
    if ds[var].ndim == 4:    
        ds = ds.isel(level=0).weighted(weights).mean(dim=("lat", "lon"))

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 4), constrained_layout=True, gridspec_kw={'width_ratios':[1, 4]})

    ax1.boxplot([ds[var].isel(time=slice(0, 17520)).values, ds[var].isel(time=slice(17520, None)).values], showfliers=False)
    ax1.set_xticklabels(['1950', '1951'])
    ax1.grid()

    ax2.plot(ds[var].isel(time=slice(0, 17520)).values, label='1950')
    ax2.plot(ds[var].isel(time=slice(17520, None)).values, label='1951')
    ax2.grid()

    # Add tick labels for 12 months
    month_indices = range(0, 17520, 17520 // 12)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ax2.set_xticks(month_indices)
    ax2.set_xticklabels(month_names, rotation=45)

    # Add title for the entire subplot
    plt.suptitle(f'Input Variable: {var}')

    plt.legend()

    if save_figure:
        plt.savefig(saving_path, format=f'{fformat}', bbox_inches='tight', pad_inches=0.1)
    
    plt.show()

def find_nearest(array, lat, lon):
    array = np.asarray(array)
    idx1 = (np.abs(array[:, 0] - lat)).argmin()
    lat = array[idx1, 0]
    
    array = array[(array[:, 0] == lat)]
    idx2 = (np.abs(array[:, 1] - lon)).argmin()
    lon = array[idx2, 1]
    return lat, lon

def plot_on_grid(df, lat, lon):

    lat, lon = find_nearest(df[['lat', 'lon']], lat, lon)

    gdf = df.groupby(['lat', 'lon'])
    g = gdf.get_group((lat, lon))
    g = g.sort_values('time')
    r2 = g['Coefficient of Determination'].iloc[0]
    print(f'exact coordinates, lat:{lat}, lon:{lon}, R2: {r2:.2f}')
    
    fig, ax = plt.subplots(figsize=(20, 4))
    g.set_index('time')[['Real Data', 'Model Output']].plot(ax=ax)
   
    plt.show()