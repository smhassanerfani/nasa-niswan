import os
import json
import warnings

import numpy as np
import xarray as xr

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


class E33OMAPAD(Dataset):

    def __init__(self, padding):
        super(E33OMAPAD, self).__init__()

        self.padding = padding

    def _padding_data(self, data):
        """
        Apply cyclic padding along the longitude (W dimension)
        and reflective padding along the latitude (H dimension).
        """
        pad_lat = self.padding[0]  # top and bottom
        pad_lon = self.padding[1]  # left and right

        H = data.shape[-2]  # latitude
        W = data.shape[-1]  # longitude

        # --- Cyclic padding along longitude (axis=4) ---
        if (pad_lon <= W):
            data = np.concatenate([data[..., -pad_lon:], data, data[..., :pad_lon]], axis=-1)
        else:
            raise AttributeError(
                f"Requested longitude padding ({pad_lon}) exceeds image width ({W})."
            )

        # --- Reflective padding along latitude (axis=3) ---
        pad_top = pad_lat
        pad_bottom = pad_lat

        if (pad_top <= H) and (pad_bottom <= H):
            top = np.flip(data[..., 1:pad_top+1, :], axis=-2)
            bottom = np.flip(data[..., -pad_bottom-1:-1, :], axis=-2)
            data = np.concatenate((top, data, bottom), axis=-2)
        else:
            raise AttributeError(
                f"Requested latitude padding ({pad_top}, {pad_bottom}) exceeds image height ({H})."
            )

        return data

    def _get_data_statistics(self):

        with open('statistics/1950.json', 'r') as jf:
            data = json.load(jf)
            data = data['set1']
        
        X1_mean = data['u']['mean']
        X2_mean = data['v']['mean']
        X3_mean = data['w']['mean']
        X4_mean = data['prec']['mean']

        X1_std = data['u']['std']
        X2_std = data['v']['std']
        X3_std = data['w']['std']
        X4_std = data['prec']['std']

        X5_mean = data['bc_src']['mean']; X5_std = data['bc_src']['std']
        self.y_mean = data['bc_conc']['mean']; self.y_std = data['bc_conc']['std']
        
        self.X_mean = np.stack([X1_mean, X2_mean, X3_mean, X4_mean, X5_mean, self.y_mean], axis=0, dtype=np.float32).reshape(1, -1, 1, 1)
        self.X_std = np.stack([X1_std, X2_std, X3_std, X4_std, X5_std, self.y_std], axis=0, dtype=np.float32).reshape(1, -1, 1, 1)


class E33OMA(E33OMAPAD):

    def __init__(self, padding, period='train', rollout_steps=1, root='/home/serfani/serfani_data1/E3OMA1950.zarr'):
        super(E33OMA, self).__init__(padding)
        
        self.root = root
        self.period = period
        self.rollout_steps = rollout_steps

        # Open as a single lazy object
        self.ds = xr.open_zarr(self.root)

        self.times = self.ds.time.values
        self.lat = self.ds.lat.values
        self.lon = self.ds.lon.values

        self._get_data_index()
        self._get_data_statistics()

    def _get_data_index(self):

        X_timestamps = self.times[:-1]
        y_timestamps = self.times[1:]

        # Create sequences
        X_datetimeindex = self.create_sequences(X_timestamps)
        y_datetimeindex = self.create_sequences(y_timestamps)

        idx = np.arange((len(self.times) * 0.5) - self.rollout_steps)  # The entire year of 1950
        rng = np.random.default_rng(42)
        rng.shuffle(idx)

        # Calculate the split point (ensuring integer)
        split_point = int(0.9 * len(idx))  # Takes the floor automatically for non-integers

        if self.period == 'train':
            self.X_datetimeindex = X_datetimeindex[idx[:split_point]]
            self.y_datetimeindex = y_datetimeindex[idx[:split_point]]
        
        elif self.period == 'val':
            self.X_datetimeindex = X_datetimeindex[idx[split_point:]] 
            self.y_datetimeindex = y_datetimeindex[idx[split_point:]]
        
        if self.period == 'test':
            self.X_datetimeindex = X_datetimeindex[len(idx):]
            self.y_datetimeindex = y_datetimeindex[len(idx):]
    
    def create_sequences(self, data_array):
        return np.array([data_array[i:i+self.rollout_steps] for i in range(len(data_array)-self.rollout_steps+1)])


    def __getitem__(self, index):

        # Get timesteps for this sample
        X_timesteps = self.X_datetimeindex[index]
        y_timesteps = self.y_datetimeindex[index]

        X = self.ds[['u', 'v', 'omega', 'prec', 'BCB_biomass_src', 'BCB']].sel(time=X_timesteps).to_array().values.transpose(1, 0, 2, 3)
        y = self.ds[['BCB']].sel(time=y_timesteps).to_array().values.transpose(1, 0, 2, 3)
        
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std

        if self.padding != [0, 0]:
            X = self._padding_data(X) # (5, seq_len, 90 + (2xpadding), 144 + (2xpadding))

        X = torch.from_numpy(X).type(torch.float32) # torch image: (channels, sequence_length, height, width)
        y = torch.from_numpy(y).type(torch.float32) # torch image: (channels, sequence_length, height, width)

        return X, y 
         
    def __len__(self):
        return len(self.X_datetimeindex)


class E33OMAModule(pl.LightningDataModule):
    def __init__(self, data_args):
        super().__init__()

        self.data_name = data_args['data_name']
        self.padding = data_args['padding']
        self.rollout_steps = data_args['sequence_length']
        self.batch_size = data_args['batch_size']
        self.num_workers = data_args['num_workers']

    def prepare_data(self):
        # This method is used to download or prepare the data. It is called only once.
        # Example: Download your dataset if it doesn't already exist in the specified directory.
        pass

    def setup(self, stage: str = None):
        # This method sets up the train, validation, and test datasets.
        # It is called on every GPU separately.
        if 'E33OMA1950' == self.data_name:
            self.train_dataset = E33OMA(period='train', padding= self.padding, species=self.species, sequence_length=self.rollout_steps)
            self.val_dataset = E33OMA(period='val',  padding= self.padding, species=self.species, sequence_length=self.rollout_steps)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True)


if __name__ == '__main__':
    
    dataset = E33OMA(period='train', padding=[83, 56], rollout_steps=3)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1, shuffle=True)

    dataiter = iter(dataloader)
    X, y = next(dataiter)

    print(len(dataset))
    print(X.shape, y.shape)

    # for i, (X, y) in enumerate(dataloader):
    #     print(i, X.shape, y.shape)