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

        ds = xr.open_dataset('/home/serfani/serfani_data1/E33OMA/ANN1950.xaijE33oma_ai.nc', decode_times=False).isel(time=0).drop_vars('time')
        
        self.X0 = ds['axyp'].values
        X0_avg = np.average(self.X0, weights=self.X0)
        X0_std = np.sqrt(np.average((self.X0  - X0_avg)**2, weights=self.X0))
        X0_avg = np.array(X0_avg, dtype=np.float32)
        X0_std = np.array(X0_std, dtype=np.float32)

        self.XL = ds['landfr'].values
        XL_avg = np.average(self.XL, weights=self.X0)
        XL_std = np.sqrt(np.average((self.XL - XL_avg)**2, weights=self.X0))
        XL_avg = np.array(XL_avg, dtype=np.float32)
        XL_std = np.array(XL_std, dtype=np.float32)

        X1_avg = np.array(data['prsurf']['avg'], dtype=np.float32)
        X2_avg = np.array(data['pblht_bp']['avg'], dtype=np.float32)
        X3_avg = np.array(data['shflx']['avg'], dtype=np.float32)
        X4_avg = np.array(data['lhflx']['avg'], dtype=np.float32)
        X5_avg = np.array(data['BCB_biomass_src']['avg'], dtype=np.float32)
        
        X6_avg = np.array(data['u']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X7_avg = np.array(data['v']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1) 
        X8_avg = np.array(data['omega']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1) 
        X9_avg = np.array(data['p_3d']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X10_avg = np.array(data['z']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X11_avg = np.array(data['t']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X12_avg = np.array(data['th']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X13_avg = np.array(data['q']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X14_avg = np.array(data['prec_3d_sum']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X15_avg = np.array(data['cfrad']['avg'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)

        self.y_mean = np.array(data['BCB']['avg'][:self.levels], dtype=np.float32).reshape(1, 1, self.levels, 1, 1)

        X1_std = np.array(data['prsurf']['std'], dtype=np.float32)
        X2_std = np.array(data['pblht_bp']['std'], dtype=np.float32)
        X3_std = np.array(data['shflx']['std'], dtype=np.float32)
        X4_std = np.array(data['lhflx']['std'], dtype=np.float32)
        X5_std = np.array(data['BCB_biomass_src']['std'], dtype=np.float32)
        
        X6_std = np.array(data['u']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X7_std = np.array(data['v']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1) 
        X8_std = np.array(data['omega']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1) 
        X9_std = np.array(data['p_3d']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X10_std = np.array(data['z']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X11_std = np.array(data['t']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X12_std = np.array(data['th']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X13_std = np.array(data['q']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X14_std = np.array(data['prec_3d_sum']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)
        X15_std = np.array(data['cfrad']['std'][:self.levels], dtype=np.float32).reshape(1, self.levels, 1, 1)

        self.y_std = np.array(data['BCB']['std'][:self.levels], dtype=np.float32).reshape(1, 1, self.levels, 1, 1)

        self.X2D_mean = np.stack([XL_avg, X0_avg, X1_avg, X2_avg, X3_avg, X4_avg, X5_avg], axis=0).reshape(1, 7, 1, 1, 1)
        self.X2D_std = np.stack([XL_std, X0_avg, X1_std, X2_std, X3_std, X4_std, X5_std], axis=0).reshape(1, 7, 1, 1, 1)

        self.X3D_mean = np.stack([X6_avg, X7_avg, X8_avg, X9_avg, X10_avg, X11_avg, X12_avg, X13_avg, X14_avg, X15_avg], axis=1) # (1, 10, levels, 1, 1)
        self.X3D_std = np.stack([X6_std, X7_std, X8_std, X9_std, X10_std, X11_std, X12_std, X13_std, X14_std, X15_std], axis=1)  # (1, 10, levels, 1, 1)    


class E33OMA(E33OMAPAD):

    def __init__(self, period, species, padding, levels=62, sequence_length=48, root='/home/serfani/serfani_data1/E3OMA1850/'):
        super(E33OMA, self).__init__(period, species, padding, levels, sequence_length)
        
        self.root = root
        self._get_data_index()
        self._get_data_statistics()
    
    def _get_data_index(self):
        
        for root, dirs, files in os.walk(self.root):
            
            sorted_files = sorted(files)
            list1 = [os.path.join(root, file) for file in sorted_files if file.split(".")[1] == 'aijh1E3oma']

        # Convert `cftime.DatetimeNoLeap` to `pandas.to_datetime()`
        warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar")
        ds = xr.open_mfdataset(list1[365:])
        datetimeindex = ds.indexes['time'].to_datetimeindex() # 52560 timestamps

        self.lat = ds.indexes['lat']
        self.lon = ds.indexes['lon']
        
        X_datetimeindex = self.create_sequences(datetimeindex) # 52560 - 48 + 1 = 52513  
        y_datetimeindex = datetimeindex[self.seq_len - 1:]

        idx = np.arange(42048 - self.seq_len + 1) # 42048 - 48 + 1 = 42001
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        idx = idx.tolist()

        if self.period == 'train':
            self.X_datetimeindex = [X_datetimeindex[i] for i in idx[:37840]] # 42001 x 0.9 = 37840
            self.datetimeindex   = [y_datetimeindex[i] for i in idx[:37840]]
        
        elif self.period == 'val':
            self.X_datetimeindex = [X_datetimeindex[i] for i in idx[37840:]] 
            self.datetimeindex   = [y_datetimeindex[i] for i in idx[37840:]]
        
        elif self.period == 'test':
            self.X_datetimeindex = X_datetimeindex[-10512:]
            self.datetimeindex   = y_datetimeindex[-10512:]

    def __getitem__(self, index):
        
        timesteps = self.X_datetimeindex[index]
        # print(timesteps)

        ls1 = [os.path.join(self.root, f'{ts}.aijh1E3oma.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds1 = xr.open_mfdataset(ls1)
        ds1['time'] = ds1.indexes['time'].to_datetimeindex()

        ls2 = [os.path.join(self.root, f'{ts}.aijlh1E3oma.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds2 = xr.open_mfdataset(ls2)
        ds2['time'] = ds2.indexes['time'].to_datetimeindex()

        ls3 = [os.path.join(self.root, f'{ts}.cijlh1E3oma.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds3 = xr.open_mfdataset(ls3)
        ds3['time'] = ds3.indexes['time'].to_datetimeindex()

        ls4 = [os.path.join(self.root, f'{ts}.taijlh1E3oma.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds4 = xr.open_mfdataset(ls4)
        ds4['time'] = ds4.indexes['time'].to_datetimeindex()

        ls5 = [os.path.join(self.root, f'{ts}.tNDaijh1E3oma.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds5 = xr.open_mfdataset(ls5)
        ds5['time'] = ds5.indexes['time'].to_datetimeindex()

        ls6 = [os.path.join(self.root, f'{ts}.rijlh1E3oma.nc') for ts in set(timesteps.strftime('%Y%m%d'))]
        ds6 = xr.open_mfdataset(ls6)
        ds6['time'] = ds6.indexes['time'].to_datetimeindex()

        XL = np.repeat(self.XL[np.newaxis, ...], self.seq_len, axis=0)
        X0 = np.repeat(self.X0[np.newaxis, ...], self.seq_len, axis=0)
        # X0 = ds1['axyp'].sel(time=slice(timesteps[0], timesteps[-1])).values
        X1 = ds1['prsurf'].sel(time=slice(timesteps[0], timesteps[-1])).values 
        X2 = ds1['pblht_bp'].sel(time=slice(timesteps[0], timesteps[-1])).values 
        X3 = ds1['shflx'].sel(time=slice(timesteps[0], timesteps[-1])).values 
        X4 = ds1['lhflx'].sel(time=slice(timesteps[0], timesteps[-1])).values 
        X5 = ds5['BCB_biomass_src'].sel(time=slice(timesteps[0], timesteps[-1])).values 
        X2D = np.stack([XL, X0, X1, X2, X3, X4, X5], axis=1) # (time, channel, lat, lon) (48, 7, 90, 144)
        X2D = X2D[:, :, np.newaxis, :, :] # (48, 7, 1, 90, 144)

        X6 = ds2['u'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X7 = ds2['v'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X8 = ds2['omega'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X9 = ds2['p_3d'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X10 = ds2['z'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X11 = ds2['t'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X12 = ds2['th'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X13 = ds2['q'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X14 = ds3['prec_3d_sum'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X15 = ds6['cfrad'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
        X3D = np.stack([X6, X7, X8, X9, X10, X11, X12, X13, X14, X15], axis=1) # (time, channel, level, lat, lon) (48, 10, 22, 90, 144)

        y = ds4['BCB'].isel(level=slice(0, self.levels)).sel(time=self.datetimeindex[index]).values
        y  = y[np.newaxis, np.newaxis, :, :, :] # (time, 1, level, lat, lon) (48, 1, 22, 90, 144)
        
        X2D = (X2D - self.X2D_mean) / self.X2D_std
        X3D = (X3D - self.X3D_mean) / self.X3D_std

        y = (y - self.y_mean) / self.y_std

        if self.padding != [0, 0]:
            X2D = self._padding_data(X2D) # (5, seq_len, 90 + (2xpadding), 144 + (2xpadding))
            X3D = self._padding_data(X3D) # (1, seq_len, 90 + (2xpadding), 144 + (2xpadding))
            
        X2D = torch.from_numpy(X2D).type(torch.float32)
        X3D = torch.from_numpy(X3D).type(torch.float32)

        y = torch.from_numpy(y).type(torch.float32) 

        return X2D, X3D, y
    
    # def __getitem__(self, index):
    #     timesteps = self.X_datetimeindex[index]
    #     date_strings = set(timesteps.strftime('%Y%m%d'))

    #     # Define file patterns and variables to load
    #     file_patterns = [
    #         ('aijh1E3oma.nc', ['axyp', 'prsurf', 'pblht_bp', 'shflx', 'lhflx']),
    #         ('aijlh1E3oma.nc', ['u', 'v', 'omega', 'p_3d', 'z', 't', 'th', 'q']),
    #         ('cijlh1E3oma.nc', ['prec_3d_sum']),
    #         ('rijlh1E3oma.nc', ['cfrad']),
    #         ('tNDaijh1E3oma.nc', ['BCB_biomass_src']),
    #         ('taijlh1E3oma.nc', ['BCB']),
    #     ]

    #     datasets = {}
    #     for pattern, variables in file_patterns:
    #         file_list = [os.path.join(self.root, f"{ts}.{pattern}") for ts in date_strings]
    #         ds = xr.open_mfdataset(file_list)
    #         ds['time'] = ds.indexes['time'].to_datetimeindex()
    #         datasets[pattern] = {var: ds[var] for var in variables}

    #     # Prepare 2D data
    #     XL = np.repeat(self.XL[np.newaxis, ...], self.seq_len, axis=0)
    #     X0 = datasets['aijh1E3oma.nc']['axyp'].sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X1 = datasets['aijh1E3oma.nc']['prsurf'].sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X2 = datasets['aijh1E3oma.nc']['pblht_bp'].sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X3 = datasets['aijh1E3oma.nc']['shflx'].sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X4 = datasets['aijh1E3oma.nc']['lhflx'].sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X5 = datasets['tNDaijh1E3oma.nc']['BCB_biomass_src'].sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X2D = np.stack([XL, X0, X1, X2, X3, X4, X5], axis=1)  # (time, channel, lat, lon)
    #     X2D = X2D[:, :, np.newaxis, :, :]  # (time, channel, 1, lat, lon)

    #     # Prepare 3D data
    #     X6 = datasets['aijlh1E3oma.nc']['u'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X7 = datasets['aijlh1E3oma.nc']['v'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X8 = datasets['aijlh1E3oma.nc']['omega'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X9 = datasets['aijlh1E3oma.nc']['p_3d'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X10 = datasets['aijlh1E3oma.nc']['z'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X11 = datasets['aijlh1E3oma.nc']['t'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X12 = datasets['aijlh1E3oma.nc']['th'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X13 = datasets['aijlh1E3oma.nc']['q'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X14 = datasets['cijlh1E3oma.nc']['prec_3d_sum'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X15 = datasets['rijlh1E3oma.nc']['cfrad'].isel(level=slice(0, self.levels)).sel(time=slice(timesteps[0], timesteps[-1])).values
    #     X3D = np.stack([X6, X7, X8, X9, X10, X11, X12, X13, X14, X15], axis=1)  # (time, channel, level, lat, lon)

    #     # Prepare target data
    #     y = datasets['taijlh1E3oma.nc']['BCB'].isel(level=slice(0, self.levels)).sel(time=self.datetimeindex[index]).values
    #     y = y[np.newaxis, np.newaxis, :, :, :]  # (1, 1, level, lat, lon)

    #     X2D = (X2D - self.X2D_mean) / self.X2D_std
    #     X3D = (X3D - self.X3D_mean) / self.X3D_std

    #     y = (y - self.y_mean) / self.y_std

    #     if self.padding != [0, 0]:
    #         X2D = self._padding_data(X2D) # (5, seq_len, 90 + (2xpadding), 144 + (2xpadding))
    #         X3D = self._padding_data(X3D) # (1, seq_len, 90 + (2xpadding), 144 + (2xpadding))
            
    #     X2D = torch.from_numpy(X2D).type(torch.float32)
    #     X3D = torch.from_numpy(X3D).type(torch.float32)

    #     y = torch.from_numpy(y).type(torch.float32) 

    #     return X2D, X3D, y


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
        self.levels = data_args['levels']
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
            self.train_dataset = E33OMA(period='train', padding= self.padding, species=self.species,
                                        levels=self.levels, sequence_length=self.seq_len)
            self.val_dataset = E33OMA(period='val',  padding= self.padding, species=self.species, 
                                      levels=self.levels, sequence_length=self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True)


if __name__ == '__main__':
    
    dataset = E33OMA(period='val', padding=[5, 5], species='bcb', levels=22, sequence_length=48)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)

    for X2D, X3D, y in dataloader:
        print(X2D.shape, X3D.shape, y.shape)
        # torch.Size([2, 48, 7, 1, 100, 154]) torch.Size([2, 48, 10, 22, 100, 154]) torch.Size([2, 1, 1, 22, 90, 144])
        # break
    # dataiter = iter(dataloader)

    # X2D, X3D, y = next(dataiter)

    # print(len(dataset))
    # print(X2D.shape, X3D.shape, y.shape)
    # torch.Size([2, 48, 7, 1, 100, 154]) torch.Size([2, 48, 10, 22, 100, 154]) torch.Size([2, 1, 1, 22, 90, 144])