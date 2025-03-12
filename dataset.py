import os
import json
import warnings

import numpy as np
import xarray as xr

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


class E33OMAPAD(Dataset):

    def __init__(self, period, species, padding, sequence_length=48):
        super(E33OMAPAD, self).__init__()

        self.period  = period
        self.species = species
        self.padding = padding
        self.seq_len = sequence_length

    def _cyclic_padding(self, data):
    
        W = data.shape[3] # longitude

        # Define the amount of padding required
        pad_left = self.padding[1]  # Padding on the left side
        pad_right = self.padding[1] # Padding on the right side

        if (pad_left <= W) and (pad_right <= W):
            
            # Cyclically extend data along the longitude
            return np.concatenate([data[..., -pad_left:], data, data[..., :pad_right]], axis=3)
        
        raise AttributeError(f"The requested padding size is larger than width size of the input image.")

    def _reflective_padding(self, data):

        H = data.shape[2] # latitude
        
        # Define the amount of padding required
        pad_top = self.padding[0]
        pad_bottom = self.padding[0]

        pad_top    += 1
        pad_bottom += 1

        if (pad_top <= H) and (pad_bottom <= H):
            
            # Reflect data at the latitude boundaries
            return np.concatenate((np.fliplr(data[:, :, 1:pad_top]), data, np.fliplr(data[:, :, -pad_bottom:-1])), axis=2)
        
        raise AttributeError(f"The requested padding size is larger than height size of the input image.")
    
    def _padding_data(self, data):
        data = self._cyclic_padding(data)
        data = self._reflective_padding(data)
        return data

    def _get_data_statistics(self):

        with open('variable_statistics.json', 'r') as jf:
            data = json.load(jf)
        
        X1_mean = data['set1']['u']['mean']
        X2_mean = data['set1']['v']['mean']
        X3_mean = data['set1']['w']['mean']
        X4_mean = data['set1']['prec']['mean']

        X1_std = data['set1']['u']['std']
        X2_std = data['set1']['v']['std']
        X3_std = data['set1']['w']['std']
        X4_std = data['set1']['prec']['std']

        if self.species == 'seasalt':
            X5_mean = data['ss_src']['mean']; X5_std = data['ss_src']['std']
            self.y_mean = data['ss_conc']['mean']; self.y_std = data['ss_conc']['std']
        
        if self.species == 'clay':
            X5_mean = data['c_src']['mean']; X5_std = data['c_src']['std']
            self.y_mean = data['c_conc']['mean']; self.y_std = data['c_conc']['std']
        
        if self.species == 'bcb':
            X5_mean = data['bc_src']['mean']; X5_std = data['bc_src']['std']
            self.y_mean = data['bc_conc']['mean']; self.y_std = data['bc_conc']['std']
        
        self.X_mean = np.array([X1_mean, X2_mean, X3_mean, X4_mean, X5_mean], dtype=np.float32).reshape(1, 5, 1, 1)
        self.X_std = np.array([X1_std, X2_std, X3_std, X4_std, X5_std], dtype=np.float32).reshape(1, 5, 1, 1)


class E33OMA(E33OMAPAD):

    def __init__(self, period, species, padding, sequence_length=48, root='/home/serfani/serfani_data1/E33OMA'):
        super(E33OMA, self).__init__(period, species, padding, sequence_length)
        
        self.root = root
        self._get_data_index()
        self._get_data_statistics()
    
    def _get_data_index(self):
        
        for root, dirs, files in os.walk(self.root):
            
            sorted_files = sorted(files)
            list1 = [os.path.join(root, file) for file in sorted_files if file.split(".")[1] == 'aijlh1E33oma_ai']   # Velocity Fields (time, level, lat, lon)

        # Convert `cftime.DatetimeNoLeap` to `pandas.to_datetime()`
        warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar")
        ds = xr.open_mfdataset(list1)
        datetimeindex = ds.indexes['time'].to_datetimeindex()  # 365 x 2 x 48 -> 35040

        self.lat = ds.indexes['lat'] 
        self.lon = ds.indexes['lon']
        
        X_datetimeindex = self.create_sequences(datetimeindex)
        y_datetimeindex = datetimeindex[self.seq_len - 1:]

        idx = np.arange(17520) # 35040 / 2 = 17520
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
        
        ls1 = [os.path.join(self.root, f'{ts}.aijlh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds1 = xr.open_mfdataset(ls1)
        ds1['time'] = ds1.indexes['time'].to_datetimeindex()

        ls2 = [os.path.join(self.root, f'{ts}.cijh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds2 = xr.open_mfdataset(ls2)
        ds2['time'] = ds2.indexes['time'].to_datetimeindex()
        
        X1 = ds1['u'].isel(level=0).sel(time=slice(timesteps[0], timesteps[-1]))
        X2 = ds1['v'].isel(level=0).sel(time=slice(timesteps[0], timesteps[-1]))
        X3 = ds1['omega'].isel(level=0).sel(time=slice(timesteps[0], timesteps[-1]))
        
        X4 = ds2['prec'].sel(time=slice(timesteps[0], timesteps[-1]))

        if self.species == 'seasalt':

            ls3 = [os.path.join(self.root, f'{ts}.taijh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
            ds3 = xr.open_mfdataset(ls3)
            ds3['time'] = ds3.indexes['time'].to_datetimeindex()

            ls4 = [os.path.join(self.root, f'{ts}.taijlh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
            ds4 = xr.open_mfdataset(ls4)
            ds4['time'] = ds4.indexes['time'].to_datetimeindex()

            X5 = ds3['seasalt1_ocean_src'].sel(time=slice(timesteps[0], timesteps[-1]))
            y  = ds4['seasalt1'].isel(level=0).sel(time=self.datetimeindex[index]).values
            y  = y[np.newaxis, np.newaxis, :, :]

        if self.species == 'clay':

            ls3 = [os.path.join(self.root, f'{ts}.tNDaijh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
            ds3 = xr.open_mfdataset(ls3)
            ds3['time'] = ds3.indexes['time'].to_datetimeindex()

            ls4 = [os.path.join(self.root, f'{ts}.taijlh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
            ds4 = xr.open_mfdataset(ls4)
            ds4['time'] = ds4.indexes['time'].to_datetimeindex()

            X5 = ds3['Clay_emission'].sel(time=slice(timesteps[0], timesteps[-1]))
            y  = ds4['Clay'].isel(level=0).sel(time=self.datetimeindex[index]).values
            y  = y[np.newaxis, np.newaxis, :, :]

        if self.species == 'bcb':

            ls3 = [os.path.join(self.root, f'{ts}.tNDaijh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
            ds3 = xr.open_mfdataset(ls3)
            ds3['time'] = ds3.indexes['time'].to_datetimeindex()

            ls4 = [os.path.join(self.root, f'{ts}.taijlh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
            ds4 = xr.open_mfdataset(ls4)
            ds4['time'] = ds4.indexes['time'].to_datetimeindex()

            X5 = ds3['BCB_biomass_src'].sel(time=slice(timesteps[0], timesteps[-1]))
            y  = ds4['BCB'].isel(level=0).sel(time=self.datetimeindex[index]).values
            y  = y[np.newaxis, np.newaxis, :, :]

        X = np.stack([X1, X2, X3, X4, X5], axis=1) # (sequence_length, channels, height, width)

        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std

        if self.padding != [0, 0]:
            X = self._padding_data(X) # (5, seq_len, 90 + (2xpadding), 144 + (2xpadding))

        X = torch.from_numpy(X).type(torch.float32) # torch image: (channels, sequence_length, height, width)
        y = torch.from_numpy(y).type(torch.float32) # torch image: (channels, sequence_length, height, width)

        return X, y 
        
    def create_sequences(self, data_array):
        sequences = [data_array[i:i+self.seq_len] for i in range(len(data_array) - self.seq_len + 1)]
        return sequences
           
    def __len__(self):
        return len(self.datetimeindex)


class E33OMA90D(E33OMAPAD):

    def __init__(self, period, species, padding, sequence_length=10, root='/home/serfani/serfani_data1/E33OMA-90Days.nc'):
        super(E33OMA90D, self).__init__(period, species, padding, sequence_length)
        self.root = root
        
        self._get_data()
    
    def _get_data(self):
        
        ds = xr.open_dataset(self.root)

        self.lat = ds.sel(lat=slice(-32, 32), lon=slice(-21.25, 56.25)).indexes['lat']
        self.lon = ds.sel(lat=slice(-32, 32), lon=slice(-21.25, 56.25)).indexes['lon']
        
        # Add negative lag for input features
        X1 = ds['u'].isel(level=0).sel(lat=slice(-32, 32), lon=slice(-21.25, 56.25)).values
        X2 = ds['v'].isel(level=0).sel(lat=slice(-32, 32), lon=slice(-21.25, 56.25)).values
        X3 = ds['omega'].isel(level=0).sel(lat=slice(-32, 32), lon=slice(-21.25, 56.25)).values
        X4 = ds['prec'].sel(lat=slice(-32, 32), lon=slice(-21.25, 56.25)).values

        if self.species == 'seasalt':
            # Add positive lag for target variable
            y  = ds['seasalt_conc'].isel(level=0).values
            X5 = ds['seasalt_src'].values
        
        if self.species == 'clay':
            y  = ds['clay_conc'].isel(level=0).values
            X5 = ds['clay_src'].values
            
        if self.species == 'bcb':
            y  = ds['bcb_conc'].isel(level=0).sel(lat=slice(-32, 32), lon=slice(-21.25, 56.25)).values
            X5 = ds['bcb_src'].sel(lat=slice(-32, 32), lon=slice(-21.25, 56.25)).values
        
        Xs = np.stack([X1, X2, X3, X4, X5], axis=1) # (sequence_length, channels, height, width)
                
        self.y_mean = y[:3023, ...].mean().reshape(-1, 1, 1)
        self.y_std = y[:3023, ...].std().reshape(-1, 1, 1)
        
        self.X_mean = Xs[:3023, ...].mean(axis=(0, 2, 3)).reshape(-1, 1, 1) 
        self.X_std = Xs[:3023, ...].std(axis=(0, 2, 3)).reshape(-1, 1, 1)

        Xs = (Xs - self.X_mean) / self.X_std
        y  = (y  - self.y_mean) / self.y_std

        X = self.create_sequences(Xs)
        y = y[self.seq_len - 1:]

        # X = np.transpose(X, (0, 2, 1, 3, 4)) # (batch_size, seq_len, 5, 90, 144) -> (batch_size, 5, seq_len, 90, 144)

        idx = np.arange(3023 + 432) # (90 x 48) = 4320
        rng = np.random.default_rng(42)
        rng.shuffle(idx)

        if self.period == "train": # 70% of the total data
            self.y = y[idx[:3023], ...]
            self.X = X[idx[:3023], ...]
            
        elif self.period == "val": # 10% of the total data
            self.y = y[idx[3023:3023+432], ...]
            self.X = X[idx[3023:3023+432], ...]
            
        elif self.period == "test": # 20% of the total data
            self.y = y[3455:, ...]
            self.X = X[3455:, ...]
            self.datetimeindex = ds.indexes['time'][3455 + self.seq_len - 1:] # Add positive lag for target variable
        
    def create_sequences(self, data_array):
        # Use sliding_window_view to create sequences
        return np.lib.stride_tricks.sliding_window_view(data_array, (self.seq_len, *data_array.shape[1:])).squeeze()

    def __getitem__(self, index):
        
        X = np.array(self.X[index, ...], copy=True)      
        y = np.array(self.y[None, None, index, ...], copy=True)

        if self.padding != [0, 0]:
            X = self._padding_data(X) # (seq_len, 5, 32 + (2xpadding), 32 + (2xpadding))
           
        X = torch.from_numpy(X).type(torch.float32) # torch image: (sequence_length, channels, height, width)
        y = torch.from_numpy(y).type(torch.float32) # torch image: (sequence_length, channels, height, width)

        return X, y # torch.Size([8, 48, 5, 32, 32]), torch.Size([8, 1, 1, 32, 32])
           
    def __len__(self):
        return len(self.y)


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
        
        elif 'E33OMA90D' == self.data_name:
            self.train_dataset = E33OMA90D(period='train', padding= self.padding, species=self.species, sequence_length=self.seq_len)
            self.val_dataset = E33OMA90D(period='val',  padding= self.padding, species=self.species, sequence_length=self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True)


if __name__ == '__main__':
    
    dataset = E33OMA90D(period='val', padding=[0, 0], species='bcb', sequence_length=48)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True)
    dataiter = iter(dataloader)

    X, y = next(dataiter)

    print(len(dataset))
    print(X.shape, y.shape)