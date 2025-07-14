import os
import json
import warnings

import numpy as np
import xarray as xr

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


class E33OMAPAD(Dataset):

    def __init__(self, period, species, padding, levels=30, sequence_length=48):
        super(E33OMAPAD, self).__init__()

        self.period  = period
        self.species = species
        self.padding = padding
        self.levels  = levels
        self.seq_len = sequence_length

    def _cyclic_padding(self, data):
    
        W = data.shape[4] # longitude

        # Define the amount of padding required
        pad_left = self.padding[1]  # Padding on the left side
        pad_right = self.padding[1] # Padding on the right side

        if (pad_left <= W) and (pad_right <= W):
            
            # Cyclically extend data along the longitude
            return np.concatenate([data[..., -pad_left:], data, data[..., :pad_right]], axis=4)
        
        raise AttributeError(f"The requested padding size is larger than width size of the input image.")

    def _reflective_padding(self, data):

        H = data.shape[3] # latitude
        
        # Define the amount of padding required
        pad_top = self.padding[0]
        pad_bottom = self.padding[0]

        pad_top    += 1
        pad_bottom += 1

        if (pad_top <= H) and (pad_bottom <= H):
            
            # Reflect data at the latitude boundaries
            return np.concatenate((np.fliplr(data[:, :, :, 1:pad_top]), data, np.fliplr(data[:, :, :, -pad_bottom:-1])), axis=3)
        
        raise AttributeError(f"The requested padding size is larger than height size of the input image.")
    
    def _padding_data(self, data):
        data = self._cyclic_padding(data)
        data = self._reflective_padding(data)
        return data

    def _get_data_statistics(self):

        with open('statistics/1850.json', 'r') as jf:
            data = json.load(jf)

        with xr.open_dataset('/home/serfani/serfani_data1/E3OMA1950/ANN1950.xaijE33oma_ai.nc', decode_times=False) as ds:
            ds = ds.isel(time=0).drop_vars('time')

            self.XA = ds['axyp'].values
            self.XL = ds['landfr'].values

        def weighted_stats(x, w):
            avg = np.average(x, weights=w)
            std = np.sqrt(np.average((x - avg)**2, weights=w))
            return np.array(avg, dtype=np.float32), np.array(std, dtype=np.float32)

        XA_avg, XA_std = weighted_stats(self.XA, self.XA)
        XL_avg, XL_std = weighted_stats(self.XL, self.XA)

        # Flat 2D vars
        self.vars_2d = ['pblht_bp', 'shflx', 'lhflx', 'BCB_biomass_src']
        X2D_avg = [XA_avg, XL_avg]
        X2D_std = [XA_std, XL_std]

        for var in self.vars_2d:
            X2D_avg.append(np.array(data[var]['avg'], dtype=np.float32))
            X2D_std.append(np.array(data[var]['std'], dtype=np.float32))

        self.X2D_mean = np.stack(X2D_avg, axis=0).reshape(1, -1, 1, 1, 1)
        self.X2D_std = np.stack(X2D_std, axis=0).reshape(1, -1, 1, 1, 1)

        # 3D vars
        self.vars_3d = ['u', 'v', 'omega', 'p_3d', 't', 'th', 'q', 'prec_3d_sum']
        X3D_avg = []
        X3D_std = []

        for var in self.vars_3d:
            avg = np.array(data[var]['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
            std = np.array(data[var]['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
            X3D_avg.append(avg)
            X3D_std.append(std)

        self.X3D_mean = np.stack(X3D_avg, axis=1)  # (1, 10, levels, 1, 1)
        self.X3D_std = np.stack(X3D_std, axis=1)

        # Target variable (y)
        # self.y_mean = np.array(data['BCB']['avg'][:self.levels], dtype=np.float32).reshape(1, 1, self.levels, 1, 1)
        # self.y_std = np.array(data['BCB']['std'][:self.levels], dtype=np.float32).reshape(1, 1, self.levels, 1, 1) 
        self.y_mean = np.array(data['BCB']['avg'][0], dtype=np.float32)
        self.y_std = np.array(data['BCB']['std'][0], dtype=np.float32)


class E3OMA(E33OMAPAD):
    """Optimized E33OMA dataset using Zarr storage"""
    
    def __init__(self, period, species, padding, levels=30, sequence_length=48, 
                 zarr_path='/home/serfani/serfani_data0/E3OMA2010D.zarr'):

        super().__init__(period, species, padding, levels, sequence_length)

        # Open zarr store
        self.ds = xr.open_zarr(zarr_path, consolidated=True)
        
        # Get coordinate information
        self.times = self.ds.time.values
        self.lat = self.ds.lat.values
        self.lon = self.ds.lon.values
        
        # Create train/val split
        self._create_data_split()

        # Load or compute statistics
        self._get_data_statistics()
    
    def _create_data_split(self):

        # Create sequences
        X_datetimeindex = self.create_sequences(self.times)
        y_datetimeindex = self.times[self.seq_len - 1:]

        idx = np.arange((2 * 365 * 48) - self.seq_len + 1)  # Total number of samples
        rng = np.random.default_rng(42)
        rng.shuffle(idx)

        # Calculate the split point (ensuring integer)
        split_point = int(0.9 * len(idx))  # Takes the floor automatically for non-integers

        if self.period == 'train':
            self.X_datetimeindex = X_datetimeindex[idx[:split_point]]
            self.datetimeindex   = y_datetimeindex[idx[:split_point]]
        
        elif self.period == 'val':
            self.X_datetimeindex = X_datetimeindex[idx[split_point:]] 
            self.datetimeindex   = y_datetimeindex[idx[split_point:]]
        
        if self.period == 'test':
            self.X_datetimeindex = X_datetimeindex[len(idx):]
            self.datetimeindex   = y_datetimeindex[len(idx):]
    
    def create_sequences(self, data_array):
        return np.array([data_array[i:i+self.seq_len] for i in range(len(data_array)-self.seq_len+1)])
    
    def __len__(self):
        return len(self.datetimeindex)
    
    def __getitem__(self, index):

        # Get timesteps for this sample
        timesteps = self.X_datetimeindex[index]
        target_time = self.datetimeindex[index]

        # Static variables
        XS = self.ds[['axyp', 'landfr']].to_array().values
        XS = np.repeat(XS[np.newaxis, :, np.newaxis, ...], self.seq_len, axis=0) # [seq_len, 2, 1, lat, lon]

        # 2D variables
        X2D = self.ds[self.vars_2d].sel(time=timesteps).to_array().values.transpose(1, 0, 2, 3)
        X2D = X2D[:, :, np.newaxis, :, :] # [seq_len, 4, 1, lat, lon]
        X2D = np.concatenate([XS, X2D], axis=1)  # [seq_len, 6, 1, lat, lon]

        # 3D variables [seq_len, 8, level, lat, lon]
        X3D = self.ds[self.vars_3d].sel(time=timesteps, level=slice(1, self.levels)).to_array().values.transpose(1, 0, 2, 3, 4)

        # Target variable
        y = self.ds['BCB'].sel(time=target_time, level=slice(1, self.levels)).values
        y = y[np.newaxis, np.newaxis, :, :]  # [1, 1, level, lat, lon]
        
        # Normalize
        X2D = (X2D - self.X2D_mean) / self.X2D_std
        X3D = (X3D - self.X3D_mean) / self.X3D_std
        y = (y - self.y_mean) / self.y_std
        
        # Apply padding if needed
        if self.padding != [0, 0]:
            X2D = self._padding_data(X2D)
            X3D = self._padding_data(X3D)
        
        # Convert to PyTorch tensors
        X2D = torch.from_numpy(X2D).float()
        X3D = torch.from_numpy(X3D).float()

        y = torch.from_numpy(y).float()

        return X2D, X3D, y

class E33OMAModule(pl.LightningDataModule):
    def __init__(self, data_args):
        super().__init__()

        self.data_name = data_args['data_name']
        self.species = data_args['species']
        self.padding = data_args['padding']
        self.levels = data_args['levels']
        self.seq_len = data_args['sequence_length']
        self.batch_size = data_args['batch_size']
        self.num_workers = data_args['num_workers']

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):

        # if 'E3OMA' == self.data_name:
        self.train_dataset = E3OMA(period='train', padding= self.padding, species=self.species,
                                    levels=self.levels, sequence_length=self.seq_len)
        self.val_dataset = E3OMA(period='val',  padding= self.padding, species=self.species, 
                                    levels=self.levels, sequence_length=self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True)


if __name__ == '__main__':
    
    dataset = E3OMA(period='val', padding=[5, 5], species='bcb', levels=20, sequence_length=48)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=True, persistent_workers=True, pin_memory=True)

    print(len(dataset))

    for X2D, X3D, y in dataloader:
        print(X2D.shape, X3D.shape, y.shape)