import os
import json
import warnings

import numpy as np
import xarray as xr

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


class E33OMAPAD(Dataset):

    def __init__(self, period, species, padding, levels=22, sequence_length=48):
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

        with open('variable_statistics.json', 'r') as jf:
            data = json.load(jf)
        
        X1_avg = np.array(data['u']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X2_avg = np.array(data['v']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1) 
        X3_avg = np.array(data['w']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1) 
        X4_avg = np.array(data['p3d']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        
        self.X5_mean = np.array(data['bc_src']['avg'], dtype=np.float32)
        self.y_mean = np.array(data['bc_conc']['avg'][:self.levels], dtype=np.float32).reshape(1, 1, self.levels, 1, 1)

        X1_std = np.array(data['u']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X2_std = np.array(data['v']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X3_std = np.array(data['w']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X4_std = np.array(data['p3d']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)

        self.X5_std = np.array(data['bc_src']['std'], dtype=np.float32)
        self.y_std = np.array(data['bc_conc']['std'][:self.levels], dtype=np.float32).reshape(1, 1, self.levels, 1, 1)

        self.X_mean = np.stack([X1_avg, X2_avg, X3_avg, X4_avg], axis=1) # (1, 4, levels, 1, 1)
        self.X_std = np.stack([X1_std, X2_std, X3_std, X4_std], axis=1)  # (1, 4, levels, 1, 1)  


class E33OMA(E33OMAPAD):

    def __init__(self, period, species, padding, levels=22, sequence_length=32, root='/home/serfani/serfani_data1/E3OMA3D-1950/'):
        super(E33OMA, self).__init__(period, species, padding, levels, sequence_length)
        
        self.root = root
        self._get_data_index()
        self._get_data_statistics()
    
    def _get_data_index(self):
        
        for root, dirs, files in os.walk(self.root):
            
            sorted_files = sorted(files)
            list1 = [os.path.join(root, file) for file in sorted_files if file.split(".")[1] == 'taijlh1E3_oma_ai_prec']   # Velocity Fields (time, level, lat, lon)

        # Convert `cftime.DatetimeNoLeap` to `pandas.to_datetime()`
        warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar")
        ds = xr.open_mfdataset(list1)
        datetimeindex = ds.indexes['time'].to_datetimeindex()  # 365 x 48 = 17520

        self.lat = ds.indexes['lat']
        self.lon = ds.indexes['lon']
        
        X_datetimeindex = self.create_sequences(datetimeindex)
        y_datetimeindex = datetimeindex[self.seq_len - 1:]

        idx = np.arange(17520)
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        idx = idx.tolist()

        if self.period == 'train':
            self.X_datetimeindex = [X_datetimeindex[i] for i in idx[:15768]] # 17520 x 0.9 = 15768
            self.datetimeindex   = [y_datetimeindex[i] for i in idx[:15768]]
        
        elif self.period == 'val':
            self.X_datetimeindex = [X_datetimeindex[i] for i in idx[15768:15768+1752]] # 17520 x 0.1 = 1752
            self.datetimeindex   = [y_datetimeindex[i] for i in idx[15768:15768+1752]]
        
        elif self.period == 'test':
            self.X_datetimeindex = X_datetimeindex[17520:] # 17520 x 0.9
            self.datetimeindex   = y_datetimeindex[17520:]

    def __getitem__(self, index):
        
        timesteps = self.X_datetimeindex[index]

        ls1 = [os.path.join(self.root, f'{ts}.aijlh1E3_oma_ai_prec.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds1 = xr.open_mfdataset(ls1)
        ds1['time'] = ds1.indexes['time'].to_datetimeindex()

        ls2 = [os.path.join(self.root, f'{ts}.cijlh1E3_oma_ai_prec.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds2 = xr.open_mfdataset(ls2)
        ds2['time'] = ds2.indexes['time'].to_datetimeindex()

        ls3 = [os.path.join(self.root, f'{ts}.tNDaijh1E3_oma_ai_prec.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds3 = xr.open_mfdataset(ls3)
        ds3['time'] = ds3.indexes['time'].to_datetimeindex()

        ls4 = [os.path.join(self.root, f'{ts}.taijlh1E3_oma_ai_prec.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds4 = xr.open_mfdataset(ls4)
        ds4['time'] = ds4.indexes['time'].to_datetimeindex()

        X1 = ds1['u'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X2 = ds1['v'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X3 = ds1['omega'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X4 = ds2['prec_3d_sum'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X = np.stack([X1, X2, X3, X4], axis=1) # (time, channel, level, lat, lon) (48, 4, 30, 90, 144)

        X5 = ds3['BCB_biomass_src'].sel(time=slice(timesteps[0], timesteps[-1])).values 
        X5 = X5[:, np.newaxis, np.newaxis, :, :] # (time, 1, 1, lat, lon) (48, 1, 1, 90, 144)

        y = ds4['BCB'].isel(level=slice(0, self.levels)).sel(time=self.datetimeindex[index]).values
        y  = y[np.newaxis, np.newaxis, :, :, :] # (time, 1, level, lat, lon) (48, 1, 22, 90, 144)
        
        X = (X - self.X_mean) / self.X_std
        X5 = (X5 - self.X5_mean) / self.X5_std

        y = (y - self.y_mean) / self.y_std

        if self.padding != [0, 0]:
            X = self._padding_data(X) # (5, seq_len, 90 + (2xpadding), 144 + (2xpadding))
            X5 = self._padding_data(X5) # (1, seq_len, 90 + (2xpadding), 144 + (2xpadding))
            
        X = torch.from_numpy(X).type(torch.float32)
        X5 = torch.from_numpy(X5).type(torch.float32)

        y = torch.from_numpy(y).type(torch.float32) 

        return X, X5, y 
        
    def create_sequences(self, data_array):
        sequences = [data_array[i:i+self.seq_len] for i in range(len(data_array) - self.seq_len + 1)]
        return sequences
           
    def __len__(self):
        return len(self.datetimeindex)


class E33OMAModule(pl.LightningDataModule):
    def __init__(self, data_args):
        super().__init__()

        self.data_name = data_args['data_name']
        self.species = data_args['species']
        self.padding = data_args['padding']
        self.seq_len = data_args['sequence_length']
        self.batch_size = data_args['batch_size']
        self.num_workers = data_args['num_workers']

    def prepare_data(self):
        # This method is used to download or prepare the data. It is called only once.
        # Example: Download your dataset if it doesn't already exist in the specified directory.
        pass

    def setup(self, stage: str = None):
        # This method sets up the train, validation, and test datasets.
        # It is called on every GPU separately.
        if 'E33OMA' == self.data_name:
            self.train_dataset = E33OMA(period='train', padding= self.padding, species=self.species, sequence_length=self.seq_len)
            self.val_dataset = E33OMA(period='val',  padding= self.padding, species=self.species, sequence_length=self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True)


if __name__ == '__main__':
    
    dataset = E33OMA(period='val', padding=[5, 5], species='bcb', levels=10, sequence_length=48)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True)
    dataiter = iter(dataloader)

    X, X5, y = next(dataiter)

    print(len(dataset))
    print(X.shape, X5.shape, y.shape)