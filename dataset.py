import os
import json
import torch
import numpy as np
import xarray as xr

from torch.utils.data import Dataset
import torchvision.transforms as T

import warnings


class E33OMA(Dataset):

    def __init__(self, period, species, padding, transform=None, root='/home/serfani/serfani_data0/E33OMA'):
        super(E33OMA, self).__init__()
        
        self.period  = period
        self.species = species
        self.padding = padding
        self.transform = transform
        self.root    = root
        
        self._get_data_index()
    
    def _get_data_index(self):
        
        for root, dirs, files in os.walk(self.root):
            
            sorted_files = sorted(files)
            list1 = [os.path.join(root, file) for file in sorted_files if file.split(".")[1] == 'aijlh1E33oma_ai']   # Velocity Fields (time, level, lat, lon)

        # Convert `cftime.DatetimeNoLeap` to `pandas.to_datetime()`
        warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar")
        datetimeindex = xr.open_mfdataset(list1[:365]).indexes['time'].to_datetimeindex()

        idx = np.arange(len(datetimeindex))
        rng = np.random.default_rng(0)
        rng.shuffle(idx)
        
        if   self.period == 'train':
            self.datetimeindex = datetimeindex[idx[:12264]]
        
        elif self.period == 'val':
            self.datetimeindex = datetimeindex[idx[12264:]]
        
        elif self.period == 'test':
            self.datetimeindex = xr.open_mfdataset(list1[365:]).indexes['time'].to_datetimeindex()

    def __getitem__(self, index):
        
        timestep = self.datetimeindex[index].strftime('%Y%m%d')
        
        ds1 = xr.open_dataset(os.path.join(self.root, f'{timestep}.aijlh1E33oma_ai.nc'))
        ds1['time'] = ds1.indexes['time'].to_datetimeindex()
        
        ds2 = xr.open_dataset(os.path.join(self.root, f'{timestep}.cijh1E33oma_ai.nc'))
        ds2['time'] = ds2.indexes['time'].to_datetimeindex()

        X1 = np.expand_dims(ds1['u'].isel(level=0).sel(time=self.datetimeindex[index]), axis=0)
        X2 = np.expand_dims(ds1['v'].isel(level=0).sel(time=self.datetimeindex[index]), axis=0)
        X3 = np.expand_dims(ds1['omega'].isel(level=0).sel(time=self.datetimeindex[index]), axis=0)

        X4 = np.expand_dims(ds2['prec'].sel(time=self.datetimeindex[index]), axis=0)

        with open('variable_statistics.json', 'r') as jf:
            data = json.load(jf)
        
            if self.transform:
                vs = data['set2']
            
            else:
                vs = data['set1']
        
        X1_mean = vs['u']['mean'];    X1_std = vs['u']['std']
        X2_mean = vs['v']['mean'];    X2_std = vs['v']['std']
        X3_mean = vs['w']['mean'];    X3_std = vs['w']['std']
        X4_mean = vs['prec']['mean']; X4_std = vs['prec']['std']

        if self.species == 'seasalt':
            # Add positive lag for target variable
            ds3 = xr.open_dataset(os.path.join(self.root, f'{timestep}.taijh1E33oma_ai.nc'))
            ds3['time'] = ds3.indexes['time'].to_datetimeindex()

            ds4 = xr.open_dataset(os.path.join(self.root, f'{timestep}.taijlh1E33oma_ai.nc'))
            ds4['time'] = ds4.indexes['time'].to_datetimeindex()

            X5 = np.expand_dims(ds3['seasalt1_ocean_src'].sel(time=self.datetimeindex[index]), axis=0)
            y  = np.expand_dims(ds4['seasalt1'].isel(level=0).sel(time=self.datetimeindex[index]), axis=0)

            X5_mean = vs['ss_src']['mean']; X5_std = vs['ss_src']['std']
            y_mean  = vs['ss_conc']['mean']; y_std = vs['ss_conc']['std']

        if self.species == 'clay':
            # Add positive lag for target variable
            ds3 = xr.open_dataset(os.path.join(self.root, f'{timestep}.tNDaijh1E33oma_ai.nc'))
            ds3['time'] = ds3.indexes['time'].to_datetimeindex()

            ds4 = xr.open_dataset(os.path.join(self.root, f'{timestep}.taijlh1E33oma_ai.nc'))
            ds4['time'] = ds4.indexes['time'].to_datetimeindex()

            X5 = np.expand_dims(ds3['Clay_emission'].sel(time=self.datetimeindex[index]), axis=0)
            y  = np.expand_dims(ds4['Clay'].isel(level=0).sel(time=self.datetimeindex[index]), axis=0)

            X5_mean = vs['c_src']['mean']; X5_std = vs['c_src']['std']
            y_mean  = vs['c_conc']['mean']; y_std = vs['c_conc']['std']

        if self.species == 'bcb':
            # Add positive lag for target variable
            ds3 = xr.open_dataset(os.path.join(self.root, f'{timestep}.tNDaijh1E33oma_ai.nc'))
            ds3['time'] = ds3.indexes['time'].to_datetimeindex()

            ds4 = xr.open_dataset(os.path.join(self.root, f'{timestep}.taijlh1E33oma_ai.nc'))
            ds4['time'] = ds4.indexes['time'].to_datetimeindex()

            X5 = np.expand_dims(ds3['BCB_biomass_src'].sel(time=self.datetimeindex[index]), axis=0)
            y  = np.expand_dims(ds4['BCB'].isel(level=0).sel(time=self.datetimeindex[index]), axis=0)

            X5_mean = vs['bc_src']['mean']; X5_std = vs['bc_src']['std']
            y_mean  = vs['bc_conc']['mean']; y_std = vs['bc_conc']['std']


        X = np.concatenate((X1, X2, X3, X4, X5), axis=0)  # (5, 90, 144)

        if self.transform:
            X = np.ma.log10(X).filled(0.0)
            y = np.ma.log10(y).filled(0.0)

        Xs_mean = np.array((X1_mean, X2_mean, X3_mean, X4_mean, X5_mean), dtype=np.float32).reshape(-1, 1, 1)
        Xs_std  = np.array((X1_std, X2_std, X3_std, X4_std, X5_std), dtype=np.float32).reshape(-1, 1, 1)
 
        self.y_mean = np.array(y_mean, dtype=np.float32).reshape(-1, 1, 1)
        self.y_std  = np.array(y_std, dtype=np.float32).reshape(-1, 1, 1)
        
        X = (X - Xs_mean) / Xs_std
        y = (y -  self.y_mean) / self.y_std

        if self.padding:
            w = X.shape[2] # width
            h = X.shape[1] # height
            
            top_pad   = self.padding - h
            right_pad = self.padding - w
            
            X = np.lib.pad(X, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        
        X = torch.from_numpy(X).type(torch.float32) # torch image: C x H x W
        y = torch.from_numpy(y).type(torch.float32) # torch image: C x H x W

        return X, y
        
    def __len__(self):
        return len(self.datetimeindex)