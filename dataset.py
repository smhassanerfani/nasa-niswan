import os
import json
import torch
import numpy as np
import xarray as xr

import os.path as osp
from torch.utils.data import Dataset
import torchvision.transforms as T

import warnings

class E33OMAPAD(Dataset):

    def __init__(self, period, species, padding):
        super(E33OMAPAD, self).__init__()
        
        self.period  = period
        self.species = species
        self.padding = padding

    def _cyclic_padding(self, data):
    
        W = data.shape[2] # longitude

        # Define the amount of padding required
        pad_left = (self.padding[1] - W) // 2  # Padding on the left side
        pad_right = self.padding[1] - W - pad_left  # Padding on the right side

        if (pad_left <= W) and (pad_right <= W):
            
            # Cyclically extend data along the longitude
            return np.concatenate([data[..., -pad_left:], data, data[..., :pad_right]], axis=2)
        
        raise AttributeError(f"The requested padding size is larger than width size of the input image.")

    def _reflective_padding(self, data):

        H = data.shape[1] # latitude
        
        # Define the amount of padding required
        pad_top = (self.padding[0] - H) // 2  # Padding on the top
        pad_bottom = self.padding[0] - H - pad_top  # Padding on the bottom

        pad_top    += 1
        pad_bottom += 1

        if (pad_top <= H) and (pad_bottom <= H):
            
            # Reflect data at the latitude boundaries
            return np.concatenate((np.fliplr(data[:, 1:pad_top]), data, np.fliplr(data[:, -pad_bottom:-1])), axis=1)
        
        raise AttributeError(f"The requested padding size is larger than height size of the input image.")

    def _padding_data(self, data):
        data = self._cyclic_padding(data)
        data = self._reflective_padding(data)
        return data


class E33OMAPADRNN(E33OMAPAD):

    def __init__(self, period, species, padding, sequence_length=10):
        super(E33OMAPADRNN, self).__init__(period, species, padding)
        self.seq_len = sequence_length

    def _cyclic_padding(self, data):
    
        W = data.shape[3] # longitude

        # Define the amount of padding required
        pad_left = (self.padding[1] - W) // 2  # Padding on the left side
        pad_right = self.padding[1] - W - pad_left  # Padding on the right side

        if (pad_left <= W) and (pad_right <= W):
            
            # Cyclically extend data along the longitude
            return np.concatenate([data[..., -pad_left:], data, data[..., :pad_right]], axis=3)
        
        raise AttributeError(f"The requested padding size is larger than width size of the input image.")

    def _reflective_padding(self, data):

        H = data.shape[2] # latitude
        
        # Define the amount of padding required
        pad_top = (self.padding[0] - H) // 2  # Padding on the top
        pad_bottom = self.padding[0] - H - pad_top  # Padding on the bottom

        pad_top    += 1
        pad_bottom += 1

        if (pad_top <= H) and (pad_bottom <= H):
            
            # Reflect data at the latitude boundaries
            return np.concatenate((np.fliplr(data[:, :, 1:pad_top]), data, np.fliplr(data[:, :, -pad_bottom:-1])), axis=2)
        
        raise AttributeError(f"The requested padding size is larger than height size of the input image.")
    
    def add_static_attributes(self):

        ds = xr.open_dataset('/home/serfani/serfani_data0/static_attrs/static_attrs.nc')

        S1 = ds['landfr'].values
        S2 = ds['ocnfr'].values
        S3 = ds['oicefr'].values

        S = np.stack([S1, S2, S3], axis=0)

        S_mean = S.mean(axis=(1, 2)).reshape(-1, 1, 1) 
        S_std  = S.std(axis=(1, 2)).reshape(-1, 1, 1)

        S = (S - S_mean) / S_std

        # Add a new axis to the array
        S = np.expand_dims(S, axis=0)

        # Replicate the array along the new axis using np.repeat
        self.S = np.repeat(S, repeats=self.seq_len, axis=0)


class E33OMA(E33OMAPAD):

    def __init__(self, period, species, padding, in_channels=5, transform=None, root='/home/serfani/serfani_data0/E33OMA'):
        super(E33OMA, self).__init__(period, species, padding)
        
        self.in_channels = in_channels
        self.transform = transform
        self.root = root
        
        self._get_data_index()
    
    def _get_data_index(self):
        
        for root, dirs, files in os.walk(self.root):
            
            sorted_files = sorted(files)
            list1 = [osp.join(root, file) for file in sorted_files if file.split(".")[1] == 'aijlh1E33oma_ai']   # Velocity Fields (time, level, lat, lon)

        # Convert `cftime.DatetimeNoLeap` to `pandas.to_datetime()`
        warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar")
        ds = xr.open_mfdataset(list1)
        datetimeindex1 = ds.indexes['time'].to_datetimeindex()[1:]  # 365 x 2 x 48 -> 35040
        datetimeindex2 = ds.indexes['time'].to_datetimeindex()[:-1] # 365 x 2 x 48 -> 35040
        self.lat = ds.indexes['lat']
        self.lon = ds.indexes['lon']

        idx = np.arange(int(35040 / 2)) # 17520
        rng = np.random.default_rng(0)
        rng.shuffle(idx)
      
        if self.period == 'train':
            self.datetimeindex1 = datetimeindex1[idx[:12264]] # 17520 x 0.7 -> 12264
            self.datetimeindex2 = datetimeindex2[idx[:12264]] # 17520 x 0.7 -> 12264
        
        elif self.period == 'val':
            self.datetimeindex1 = datetimeindex1[idx[12264:]] # 17520 x 0.3
            self.datetimeindex2 = datetimeindex2[idx[12264:]] # 17520 x 0.3
        
        elif self.period == 'test':
            self.datetimeindex1 = datetimeindex1[17520:] # 17520 - 1
            self.datetimeindex2 = datetimeindex2[17520:] # 17520 - 1

    def __getitem__(self, index):
        
        timestep1 = self.datetimeindex1[index].strftime('%Y%m%d')
        timestep2 = self.datetimeindex2[index].strftime('%Y%m%d')
        
        ds1 = xr.open_dataset(osp.join(self.root, f'{timestep1}.aijlh1E33oma_ai.nc'))
        ds1['time'] = ds1.indexes['time'].to_datetimeindex()
        
        ds2 = xr.open_dataset(osp.join(self.root, f'{timestep1}.cijh1E33oma_ai.nc'))
        ds2['time'] = ds2.indexes['time'].to_datetimeindex()

        X1 = np.expand_dims(ds1['u'].isel(level=0).sel(time=self.datetimeindex1[index]), axis=0)
        X2 = np.expand_dims(ds1['v'].isel(level=0).sel(time=self.datetimeindex1[index]), axis=0)
        X3 = np.expand_dims(ds1['omega'].isel(level=0).sel(time=self.datetimeindex1[index]), axis=0)

        X4 = np.expand_dims(ds2['prec'].sel(time=self.datetimeindex1[index]), axis=0)

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
            ds3 = xr.open_dataset(osp.join(self.root, f'{timestep1}.taijh1E33oma_ai.nc'))
            ds3['time'] = ds3.indexes['time'].to_datetimeindex()

            ds4 = xr.open_dataset(osp.join(self.root, f'{timestep1}.taijlh1E33oma_ai.nc'))
            ds4['time'] = ds4.indexes['time'].to_datetimeindex()

            ds5 = xr.open_dataset(osp.join(self.root, f'{timestep2}.taijlh1E33oma_ai.nc'))
            ds5['time'] = ds5.indexes['time'].to_datetimeindex()

            X5 = np.expand_dims(ds3['seasalt1_ocean_src'].sel(time=self.datetimeindex1[index]), axis=0)
            y  = np.expand_dims(ds4['seasalt1'].isel(level=0).sel(time=self.datetimeindex1[index]), axis=0)

            X6 = np.expand_dims(ds5['seasalt1'].isel(level=0).sel(time=self.datetimeindex2[index]), axis=0)

            X5_mean = vs['ss_src']['mean']; X5_std = vs['ss_src']['std']
            y_mean  = vs['ss_conc']['mean']; y_std = vs['ss_conc']['std']

        if self.species == 'clay':
            # Add positive lag for target variable
            ds3 = xr.open_dataset(osp.join(self.root, f'{timestep1}.tNDaijh1E33oma_ai.nc'))
            ds3['time'] = ds3.indexes['time'].to_datetimeindex()

            ds4 = xr.open_dataset(osp.join(self.root, f'{timestep1}.taijlh1E33oma_ai.nc'))
            ds4['time'] = ds4.indexes['time'].to_datetimeindex()

            ds5 = xr.open_dataset(osp.join(self.root, f'{timestep2}.taijlh1E33oma_ai.nc'))
            ds5['time'] = ds5.indexes['time'].to_datetimeindex()

            X5 = np.expand_dims(ds3['Clay_emission'].sel(time=self.datetimeindex1[index]), axis=0)
            y  = np.expand_dims(ds4['Clay'].isel(level=0).sel(time=self.datetimeindex1[index]), axis=0)

            X6 = np.expand_dims(ds5['Clay'].isel(level=0).sel(time=self.datetimeindex2[index]), axis=0)

            X5_mean = vs['c_src']['mean']; X5_std = vs['c_src']['std']
            y_mean  = vs['c_conc']['mean']; y_std = vs['c_conc']['std']

        if self.species == 'bcb':
            # Add positive lag for target variable
            ds3 = xr.open_dataset(osp.join(self.root, f'{timestep1}.tNDaijh1E33oma_ai.nc'))
            ds3['time'] = ds3.indexes['time'].to_datetimeindex()

            ds4 = xr.open_dataset(osp.join(self.root, f'{timestep1}.taijlh1E33oma_ai.nc'))
            ds4['time'] = ds4.indexes['time'].to_datetimeindex()

            ds5 = xr.open_dataset(osp.join(self.root, f'{timestep2}.taijlh1E33oma_ai.nc'))
            ds5['time'] = ds5.indexes['time'].to_datetimeindex()

            X5 = np.expand_dims(ds3['BCB_biomass_src'].sel(time=self.datetimeindex1[index]), axis=0)
            y  = np.expand_dims(ds4['BCB'].isel(level=0).sel(time=self.datetimeindex1[index]), axis=0)

            X6 = np.expand_dims(ds5['BCB'].isel(level=0).sel(time=self.datetimeindex2[index]), axis=0)

            X5_mean = vs['bc_src']['mean']; X5_std = vs['bc_src']['std']
            y_mean  = vs['bc_conc']['mean']; y_std = vs['bc_conc']['std']

        if self.in_channels == 5:
            X = np.concatenate((X1, X2, X3, X4, X5), axis=0)  # (5, 90, 144)

            if self.transform:
                X = np.ma.log10(X).filled(0.0)
                y = np.ma.log10(y).filled(0.0)

            Xs_mean = np.array((X1_mean, X2_mean, X3_mean, X4_mean, X5_mean), dtype=np.float32).reshape(-1, 1, 1)
            Xs_std  = np.array((X1_std, X2_std, X3_std, X4_std, X5_std), dtype=np.float32).reshape(-1, 1, 1)
    
            self.y_mean = np.array(y_mean, dtype=np.float32).reshape(-1, 1, 1)
            self.y_std  = np.array(y_std, dtype=np.float32).reshape(-1, 1, 1)
        
        elif self.in_channels == 6:
            X = np.concatenate((X1, X2, X3, X4, X5, X6), axis=0)  # (6, 90, 144)

            if self.transform:
                X = np.ma.log10(X).filled(0.0)
                y = np.ma.log10(y).filled(0.0)

            Xs_mean = np.array((X1_mean, X2_mean, X3_mean, X4_mean, X5_mean, y_mean), dtype=np.float32).reshape(-1, 1, 1)
            Xs_std  = np.array((X1_std, X2_std, X3_std, X4_std, X5_std, y_std), dtype=np.float32).reshape(-1, 1, 1)
    
            self.y_mean = np.array(y_mean, dtype=np.float32).reshape(-1, 1, 1)
            self.y_std  = np.array(y_std, dtype=np.float32).reshape(-1, 1, 1)
            
        X = (X - Xs_mean) / Xs_std
        y = (y -  self.y_mean) / self.y_std

        if self.padding: # torch image: C x H x W -> (5, 256, 256)
            X = self._padding_data(X)
        
        X = torch.from_numpy(X).type(torch.float32) # torch image: C x H x W
        y = torch.from_numpy(y).type(torch.float32) # torch image: C x H x W

        return X, y
       
    def __getattr__(self, name):
        if name == 'datetimeindex':
            return self.datetimeindex1
        raise AttributeError(f"'Library' object has no attribute '{name}'")

    def __len__(self):
        return len(self.datetimeindex)


class E33OMA90D(E33OMAPAD):

    def __init__(self, period, species, padding, in_channels=5, transform=None, root='/home/serfani/serfani_data0/E33OMA-90Days.nc'):
        super(E33OMA90D, self).__init__(period, species, padding)
        
        self.in_channels = in_channels
        self.transform = transform
        self.root = root
        
        self._get_data()
    
    
    def _get_data(self):
        
        ds = xr.open_dataset(self.root)

        self.lat = ds.indexes['lat']
        self.lon = ds.indexes['lon']
        
        # Add negative lag for input features
        X1 = np.expand_dims(ds['u'].isel(level=0)[1:, ...], axis=1)
        X2 = np.expand_dims(ds['v'].isel(level=0)[1:, ...], axis=1)
        X3 = np.expand_dims(ds['omega'].isel(level=0)[1:, ...], axis=1)
        X4 = np.expand_dims(ds['prec'][1:, ...], axis=1)

        if self.species == 'seasalt':
            # Add positive lag for target variable
            y  = np.expand_dims(ds['seasalt_conc'].isel(level=0)[1:, ...], axis=1) # (4319, 1, 90, 144)
            X5 = np.expand_dims(ds['seasalt_src'][1:, ...], axis=1) # (4319, 1, 90, 144)
            X6 = np.expand_dims(ds['seasalt_conc'].isel(level=0)[:-1, ...], axis=1) # (4319, 1, 90, 144) # previous time step
        
        if self.species == 'clay':
            y  = np.expand_dims(ds['clay_conc'].isel(level=0)[1:, ...], axis=1)
            X5 = np.expand_dims(ds['clay_src'][1:, ...], axis=1)
            X6 = np.expand_dims(ds['clay_conc'].isel(level=0)[:-1, ...], axis=1)
            
        if self.species == 'bcb':
            y  = np.expand_dims(ds['bcb_conc'].isel(level=0)[1:, ...], axis=1)
            X5 = np.expand_dims(ds['bcb_src'][1:, ...], axis=1)
            X6 = np.expand_dims(ds['bcb_conc'].isel(level=0)[:-1, ...], axis=1)
        
        idx = np.arange(3455) # train + validation
        
        rng = np.random.default_rng(0)
        rng.shuffle(idx)
        
        if self.transform: 
            
            y  = np.ma.log10(y).filled(0.0)
            
            X1 = np.ma.log10(X1).filled(0.0)
            X2 = np.ma.log10(X2).filled(0.0)
            X3 = np.ma.log10(X3).filled(0.0)
            X4 = np.ma.log10(X4).filled(0.0)
            X5 = np.ma.log10(X5).filled(0.0)
            X6 = np.ma.log10(X6).filled(0.0)
        
        if self.in_channels == 5:
            X = np.concatenate((X1, X2, X3, X4, X5), axis=1) # (4319, 5, 90, 144)
        
        if self.in_channels == 6:  
            X = np.concatenate((X1, X2, X3, X4, X5, X6), axis=1) # (4319, 6, 90, 144)
                
        self.y_mean = y[idx[:3023], ...].mean().reshape(-1, 1, 1)
        self.y_std  = y[idx[:3023], ...].std().reshape(-1, 1, 1)
        
        self.X_mean = X[idx[:3023], ...].mean(axis=(0, 2, 3)).reshape(-1, 1, 1) 
        self.X_std  = X[idx[:3023], ...].std(axis=(0, 2, 3)).reshape(-1, 1, 1)

        if self.period == "train": # 70% of the total data
            self.y = y[idx[:3023], ...]
            self.X = X[idx[:3023], ...]
            
        elif self.period == "val": # 10% of the total data
            self.y = y[idx[3023:3455], ...]
            self.X = X[idx[3023:3455], ...]
            
        elif self.period == "test": # 20% of the total data
            self.y = y[3455:, ...]
            self.X = X[3455:, ...]
            self.datetimeindex = ds.indexes['time'][3455 + 1:] # Add positive lag for target variable
        
    def __getitem__(self, index):
        
        X = self.X[index, ...]
        y = self.y[index, ...]
    
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std
               
        if self.padding: # torch image: C x H x W -> (5, 255, 255)
            X = self._padding_data(X)
        
        X = torch.from_numpy(X).type(torch.float32) # torch image: C x H x W
        y = torch.from_numpy(y).type(torch.float32) # torch image: C x H x W

        return X, y
           
    def __len__(self):
        return len(self.y)


class E33OMA_CRNN(E33OMAPADRNN):

    def __init__(self, period, species, padding, in_channels=5, sequence_length=10, root='/home/serfani/serfani_data0/E33OMA'):
        super(E33OMA_CRNN, self).__init__(period, species, padding, sequence_length)
        
        self.in_channels = in_channels
        self.root = root
        
        self._get_data_index()
    
    
    def _get_data_index(self):
        
        for root, dirs, files in os.walk(self.root):
            
            sorted_files = sorted(files)
            list1 = [osp.join(root, file) for file in sorted_files if file.split(".")[1] == 'aijlh1E33oma_ai']   # Velocity Fields (time, level, lat, lon)

        # Convert `cftime.DatetimeNoLeap` to `pandas.to_datetime()`
        warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar")
        ds = xr.open_mfdataset(list1)
        datetimeindex = ds.indexes['time'].to_datetimeindex()  # 365 x 2 x 48 -> 35040

        self.lat = ds.indexes['lat']
        self.lon = ds.indexes['lon']
        
        X_datetimeindex = self.create_sequences(datetimeindex)
        y_datetimeindex = datetimeindex[self.seq_len - 1:]
        
        if self.period == 'train':
            self.X_datetimeindex = X_datetimeindex[:17520] # 17520 x 1.0 (1950)
            self.datetimeindex   = y_datetimeindex[:17520] 
        
        elif self.period == 'val':
            self.X_datetimeindex = X_datetimeindex[17520:17520+1752] # 17520 x 0.1 (1951)
            self.datetimeindex   = y_datetimeindex[17520:17520+1752] 
        
        elif self.period == 'test':
            self.X_datetimeindex = X_datetimeindex[17520+1752:] # 17520 x 0.9
            self.datetimeindex   = y_datetimeindex[17520+1752:]

    def __getitem__(self, index):
        
        timesteps = self.X_datetimeindex[index]
        
        ls1 = [osp.join(self.root, f'{ts}.aijlh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds1 = xr.open_mfdataset(ls1)
        ds1['time'] = ds1.indexes['time'].to_datetimeindex()

        ls2 = [osp.join(self.root, f'{ts}.cijh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds2 = xr.open_mfdataset(ls2)
        ds2['time'] = ds2.indexes['time'].to_datetimeindex()
        
        X1 = ds1['u'].isel(level=0).sel(time=slice(timesteps[0], timesteps[-1]))
        X2 = ds1['v'].isel(level=0).sel(time=slice(timesteps[0], timesteps[-1]))
        X3 = ds1['omega'].isel(level=0).sel(time=slice(timesteps[0], timesteps[-1]))
        
        X4 = ds2['prec'].sel(time=slice(timesteps[0], timesteps[-1]))

        with open('variable_statistics.json', 'r') as jf:
            data = json.load(jf)
            vs = data['set1']
        
        X1_mean = vs['u']['mean'];    X1_std = vs['u']['std']
        X2_mean = vs['v']['mean'];    X2_std = vs['v']['std']
        X3_mean = vs['w']['mean'];    X3_std = vs['w']['std']
        X4_mean = vs['prec']['mean']; X4_std = vs['prec']['std']

        if self.species == 'seasalt':

            ls3 = [osp.join(self.root, f'{ts}.taijh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
            ds3 = xr.open_mfdataset(ls3)
            ds3['time'] = ds3.indexes['time'].to_datetimeindex()

            ls4 = [osp.join(self.root, f'{ts}.taijlh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
            ds4 = xr.open_mfdataset(ls4)
            ds4['time'] = ds4.indexes['time'].to_datetimeindex()

            X5 = ds3['seasalt1_ocean_src'].sel(time=slice(timesteps[0], timesteps[-1]))
            y  = ds4['seasalt1'].isel(level=0).sel(time=self.datetimeindex[index]).values

            X5_mean = vs['ss_src']['mean']; X5_std = vs['ss_src']['std']
            y_mean  = vs['ss_conc']['mean']; y_std = vs['ss_conc']['std']

        if self.species == 'clay':

            ls3 = [osp.join(self.root, f'{ts}.tNDaijh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
            ds3 = xr.open_mfdataset(ls3)
            ds3['time'] = ds3.indexes['time'].to_datetimeindex()

            ls4 = [osp.join(self.root, f'{ts}.taijlh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
            ds4 = xr.open_mfdataset(ls4)
            ds4['time'] = ds4.indexes['time'].to_datetimeindex()

            X5 = ds3['Clay_emission'].sel(time=slice(timesteps[0], timesteps[-1]))
            y  = ds4['Clay'].isel(level=0).sel(time=self.datetimeindex[index]).values

            X5_mean = vs['c_src']['mean']; X5_std = vs['c_src']['std']
            y_mean  = vs['c_conc']['mean']; y_std = vs['c_conc']['std']

        if self.species == 'bcb':

            ls3 = [osp.join(self.root, f'{ts}.tNDaijh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
            ds3 = xr.open_mfdataset(ls3)
            ds3['time'] = ds3.indexes['time'].to_datetimeindex()

            ls4 = [osp.join(self.root, f'{ts}.taijlh1E33oma_ai.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
            ds4 = xr.open_mfdataset(ls4)
            ds4['time'] = ds4.indexes['time'].to_datetimeindex()

            X5 = ds3['BCB_biomass_src'].sel(time=slice(timesteps[0], timesteps[-1]))
            y  = ds4['BCB'].isel(level=0).sel(time=self.datetimeindex[index]).values

            X5_mean = vs['bc_src']['mean']; X5_std = vs['bc_src']['std']
            y_mean  = vs['bc_conc']['mean']; y_std = vs['bc_conc']['std']


        X_means = np.array((X1_mean, X2_mean, X3_mean, X4_mean, X5_mean), dtype=np.float32).reshape(1, 5, 1, 1)
        X_stds  = np.array((X1_std, X2_std, X3_std, X4_std, X5_std), dtype=np.float32).reshape(1, 5, 1, 1)

        self.y_mean = np.array(y_mean, dtype=np.float32)
        self.y_std  = np.array(y_std, dtype=np.float32)

        X = np.stack([X1, X2, X3, X4, X5], axis=1) 

        X = (X - X_means) / X_stds
        y = (y -  self.y_mean) / self.y_std

        if self.in_channels == 8:
            self.add_static_attributes()
            X = np.concatenate((X, self.S), axis=1)

        if self.padding:
            X = self._padding_data(X) # (seq_len, 5, 90 + (2xpadding), 144 + (2xpadding))

        X = torch.from_numpy(X).type(torch.float32) # torch image: (sequence_length, channels, height, width)
        y = torch.from_numpy(y).type(torch.float32) # torch image: (sequence_length, channels, height, width)

        return X, y 
        
    def create_sequences(self, data_array):
        sequences = [data_array[i:i+self.seq_len] for i in range(len(data_array) - self.seq_len + 1)]
        return sequences
           
    def __len__(self):
        return len(self.datetimeindex)


class E33OMA90D_CRNN(E33OMAPADRNN):

    def __init__(self, period, species, padding, in_channels=5, sequence_length=10, root='/home/serfani/serfani_data0/E33OMA-90Days.nc'):
        super(E33OMA90D_CRNN, self).__init__(period, species, padding, sequence_length)
        
        self.in_channels = in_channels
        self.root = root
        
        self._get_data()
    
    def _get_data(self):
        
        ds = xr.open_dataset(self.root)

        self.lat = ds.indexes['lat']
        self.lon = ds.indexes['lon']
        
        # Add negative lag for input features
        X1 = ds['u'].isel(level=0).values
        X2 = ds['v'].isel(level=0).values
        X3 = ds['omega'].isel(level=0).values
        X4 = ds['prec'].values

        if self.species == 'seasalt':
            # Add positive lag for target variable
            y  = ds['seasalt_conc'].isel(level=0).values
            X5 = ds['seasalt_src'].values
        
        if self.species == 'clay':
            y  = ds['clay_conc'].isel(level=0).values
            X5 = ds['clay_src'].values
            
        if self.species == 'bcb':
            y  = ds['bcb_conc'].isel(level=0).values
            X5 = ds['bcb_src'].values
        
        Xs = np.stack([X1, X2, X3, X4, X5], axis=1)
                
        self.y_mean = y[:3023, ...].mean().reshape(-1, 1, 1)
        self.y_std  = y[:3023, ...].std().reshape(-1, 1, 1)
        
        self.X_mean = Xs[:3023, ...].mean(axis=(0, 2, 3)).reshape(-1, 1, 1) 
        self.X_std  = Xs[:3023, ...].std(axis=(0, 2, 3)).reshape(-1, 1, 1)

        Xs = (Xs - self.X_mean) / self.X_std
        y  = (y  - self.y_mean) / self.y_std

        X = self.create_sequences(Xs)
        y = y[self.seq_len - 1:]

        if self.period == "train": # 70% of the total data
            self.y = y[:3023, ...]
            self.X = X[:3023, ...]
            
        elif self.period == "val": # 10% of the total data
            self.y = y[3023:3455, ...]
            self.X = X[3023:3455, ...]
            
        elif self.period == "test": # 20% of the total data
            self.y = y[3455:, ...]
            self.X = X[3455:, ...]
            self.datetimeindex = ds.indexes['time'][3455 + self.seq_len - 1:] # Add positive lag for target variable
        
    def create_sequences(self, data_array):
        # Use sliding_window_view to create sequences
        return np.lib.stride_tricks.sliding_window_view(data_array, (self.seq_len, *data_array.shape[1:])).squeeze()

    def __getitem__(self, index):
        
        X = np.array(self.X[index, ...], copy=True)
        
        if self.in_channels == 8:
            self.add_static_attributes()
            X = np.concatenate((X, self.S), axis=1)
        
        y = np.array(self.y[index, ...], copy=True)

        if self.padding:
            X = self._padding_data(X) # (seq_len, 5, 90 + (2xpadding), 144 + (2xpadding))
           
        X = torch.from_numpy(X).type(torch.float32) # torch image: (sequence_length, channels, height, width)
        y = torch.from_numpy(y).type(torch.float32) # torch image: (sequence_length, channels, height, width)

        return X, y
           
    def __len__(self):
        return len(self.y)

if __name__ == '__main__':
    
    # dataset = E33OMA(period='test', species='bcb', padding=(256, 256), in_channels=6, transform=None)
    dataset = E33OMA_CRNN(period='test', padding=(100, 154), species='bcb', in_channels=8, sequence_length=15)
    dataiter = iter(dataset)
    
    X, y = next(dataiter)
    
    print(len(dataset))
    print(X.shape, y.shape)
    print(dataset.datetimeindex)