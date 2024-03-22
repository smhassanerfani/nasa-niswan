import os
import torch
import numpy as np
import xarray as xr

from torch.utils.data import Dataset
import torchvision.transforms as T

class E33OMA(Dataset):

    def __init__(self, split, padding=None, joint_transform=None):
        super(E33OMA, self).__init__()
        
        self.split = split
        self.padding = padding
        self.joint_transform = joint_transform
        self.transform = T.ToTensor()
        
        self._get_data()
    
    
    def _get_data(self):
        
        for root, dirs, files in os.walk('/discover/nobackup/sebauer1/E33oma_ai/output/'):

            sorted_files = sorted(files)
            ds_list1 = [os.path.join(root, file) for file in sorted_files if file.split(".")[1] == 'aijlh1E33oma_ai']
            ds_list2 = [os.path.join(root, file) for file in sorted_files if file.split(".")[1] == 'taijlh1E33oma_ai']
            ds_list3 = [os.path.join(root, file) for file in sorted_files if file.split(".")[1] in ['cijh1E33oma_ai', 'taijh1E33oma_ai']]

        ds1 = xr.open_mfdataset(ds_list1)
        ds1 = ds1.isel(level=0).drop_vars('level')
        ds1 = ds1.drop_vars(['axyp'])

        ds2 = xr.open_mfdataset(ds_list2)
        ds2 = ds2.isel(level=0).drop_vars('level')
        ds2 = ds2.drop_vars(['axyp', 'Clay', 'BCB'])

        ds3 = xr.open_mfdataset(ds_list3)

        ds1['seasalt'] = ds2['seasalt1']
        ds1['prec'] = ds3['prec']
        ds1['seasalt_src'] = ds3['seasalt1_ocean_src']

        datetimeindex = ds1.indexes['time'].to_datetimeindex()
        ds1['time'] = datetimeindex
        
        ds1.load()

        # Add positive lag for target variable
        target = np.expand_dims(ds1['seasalt'][:, ...], axis=1) # (4319, 1, 90, 144) [1:, ...]

        # Add negative lag for input features
        u = np.expand_dims(ds1['u'][:, ...], axis=1) # (4319, 1, 90, 144) [:-1, ...]
        v = np.expand_dims(ds1['v'][:, ...], axis=1) # (4319, 1, 90, 144)
        omega = np.expand_dims(ds1['omega'][:, ...], axis=1) # (4319, 1, 90, 144)
        prec = np.expand_dims(ds1['prec'][:, ...], axis=1)   # (4319, 1, 90, 144)
        seasalt_src = np.expand_dims(ds1['seasalt_src'][:, ...], axis=1) # (4319, 1, 90, 144)

        features = np.concatenate((u, v, omega, prec, seasalt_src), axis=1) # (4319, 5, 90, 144)

        
        self.target_min = target[:3023, ...].min().reshape(-1, 1, 1)
        self.target_max = target[:3023, ...].max().reshape(-1, 1, 1)

        self.features_min = features[:3023, ...].min(axis=(0, 2, 3)).reshape(-1, 1, 1) 
        self.features_max = features[:3023, ...].max(axis=(0, 2, 3)).reshape(-1, 1, 1)
        
        if self.split == "train": # 70% of the total data
            self.target = target[:3023, ...]
            self.features = features[:3023, ...]
        
        elif self.split == "val": # 10% of the total data
            self.target = target[3023:3455, ...]
            self.features = features[3023:3455, ...]
            
        else: # (self.split == "test") # 20% of the total data
            self.target = target[3455:, ...]
            self.features = features[3455:, ...]
        
    def __getitem__(self, index):
        
        X = self.features[index, ...]
        y = self.target[index, ...]
    
        X = (X - self.features_min) / (self.features_max - self.features_min)
        y = (y - self.target_min)   / (self.target_max - self.target_min)

        X = 2 * X - 1
        y = 2 * y - 1
        
        if self.padding:
            w = X.shape[2] # width
            h = X.shape[1] # height
            
            top_pad   = self.padding - h
            right_pad = self.padding - w
            
            X = np.lib.pad(X, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        
        X = torch.from_numpy(X) # torch image: C x H x W -> (5, 256, 256)
        y = torch.from_numpy(y) # torch image: C x H x W -> (1, 90, 144)
            
        if self.joint_transform:
            X, y = self.joint_transform(X, y)

        return X, y
        
    def __len__(self):
        return len(self.target)