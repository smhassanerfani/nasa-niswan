import os
import warnings
import xarray as xr
import pandas as pd
import numpy as np
import zarr
from datetime import datetime
import dask
from dask.distributed import Client, LocalCluster
import shutil

def convert_e33oma_to_zarr(netcdf_root, zarr_path, use_dask=True, n_workers=4):
    """
    Convert E33OMA NetCDF files to a unified Zarr store
    
    Parameters:
    -----------
    netcdf_root : str
        Path to NetCDF files
    zarr_path : str
        Output path for Zarr store
    use_dask : bool
        Whether to use Dask for parallel processing
    n_workers : int
        Number of Dask workers to use
    """
    # Setup Dask if requested
    client = None
    if use_dask:
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)
        print(f"Dask dashboard: {client.dashboard_link}")
    
    # Create zarr directory if it doesn't exist
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
    
    # If zarr store exists, remove it to start fresh
    if os.path.exists(zarr_path):
        print(f"Removing existing zarr store at {zarr_path}")
        shutil.rmtree(zarr_path)
    
    # Find unique dates
    all_dates = list()
    for root, dirs, files in os.walk(netcdf_root):
        for file in sorted(files):
            if file.endswith('.aijh1E3oma.nc'):
                date = file.split('.')[0]
                all_dates.append(date)

    all_dates = sorted(all_dates)
    print(f"Found {len(all_dates)} unique dates")
    
    # Process variables in separate groups for more efficient access
    process_variable_group(netcdf_root, zarr_path, all_dates, 'aijh', 
                         ['pblht_bp', 'shflx', 'lhflx'], client)
    
    process_variable_group(netcdf_root, zarr_path, all_dates, 'aijlh', 
                         ['u', 'v', 'omega'], client, level=30)

    process_variable_group(netcdf_root, zarr_path, all_dates, 'aijlh', 
                     ['t', 'th', 'q', 'p_3d'], client, level=30)
    
    process_variable_group(netcdf_root, zarr_path, all_dates, 'cijlh', 
                         ['prec_3d_sum'], client, level=30)
    
    process_variable_group(netcdf_root, zarr_path, all_dates, 'tNDaijh', 
                         ['BCB_biomass_src'], client)
    
    process_variable_group(netcdf_root, zarr_path, all_dates, 'taijlh', 
                         ['BCB'], client, level=30)
    
    add_grid_info(zarr_path)

    # Close Dask client
    if client:
        client.close()
    
    print(f"Zarr store created at {zarr_path}")
    print("Creating time index file...")
    print("Conversion complete!")
    
def process_variable_group(netcdf_root, zarr_path, all_dates, file_type, variables, client=None, level=None):
    
    # Filter the specific warning about to_datetimeindex before calling the function
    warnings.filterwarnings("ignore", message="In a future version of xarray to_datetimeindex will default to returning")
    warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar")
    warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")

    """Process a group of variables from the same file type"""
    print(f"Processing {file_type} variables: {variables}")
    
    # Create file lists for each date
    file_list = [os.path.join(netcdf_root, f"{date}.{file_type}1E3oma.nc") for date in all_dates]
    
    # Define optimal chunking strategy - adjust based on your data size
    chunks = {
        'time': 48,        # One day of hourly data
        'lat': 90,         # Half of typical 90 latitude points
        'lon': 144,         # Half of typical 144 longitude points
        'level': 30 if level is not None else 1         # Process one level at a time if applicable
    }

    # Open dataset with dask chunks
    ds = xr.open_mfdataset(
        file_list, 
        combine='nested',
        concat_dim='time',
        chunks=chunks,
        parallel=True,
        engine='netcdf4'
    )
    
    # Ensure time is a standard pandas DatetimeIndex
    if not isinstance(ds.indexes['time'], pd.DatetimeIndex):
        ds['time'] = ds.indexes['time'].to_datetimeindex()
    
    # Create subset with only needed variables
    subset = ds[variables]
    
    # If level is specified, select that level
    if level is not None and 'level' in subset.dims:
        subset = subset.sel(level=slice(1, level))
    
    for var in subset.data_vars:
        print(var, subset[var].shape, subset[var].dtype)


   # Apply simple chunking without explicit compressor
    encoding = {}
    for var in subset.data_vars:
        var_chunks = {}
        dims = subset[var].dims
        
        # Set chunk sizes based on dimension names
        if 'time' in dims:
            var_chunks['time'] = 48
        if 'level' in dims:
            var_chunks['level'] = 30
        if 'lat' in dims:
            var_chunks['lat'] = 90
        if 'lon' in dims:
            var_chunks['lon'] = 144
            
        encoding[var] = {'chunks': var_chunks}
    print("Encoding configuration:", encoding)

    # Write to zarr
    print(f"Writing {file_type} variables to zarr...")
    subset.to_zarr(
        zarr_path, 
        mode='a',  # Append to existing store
        # encoding=encoding,
        consolidated=True,
        compute=True,  # Wait for computation to finish
        safe_chunks=True  # Use safe chunking
    )
        
    # Close dataset to free memory
    ds.close()

def add_grid_info(zarr_path):
    """Add 2D grid variables to the zarr store"""
    print("Adding 2D grid information variables...")
    
    # Load 2D grid file - assuming it contains axyp and landfr variables
    grid_file = '/home/serfani/serfani_data1/E3OMA1950/ANN1950.xaijE33oma_ai.nc'
    ds_2d = xr.open_dataset(grid_file, decode_times=False).isel(time=0).drop_vars('time')
    
    # Select only the grid variables
    grid_vars = ds_2d[['axyp', 'landfr']]
    
    # Set up encoding for 2D variables
    encoding = {}
    for var in grid_vars.data_vars:
        var_chunks = {}
        dims = grid_vars[var].dims
        
        # Set chunk sizes based on dimension names
        if 'lat' in dims:
            var_chunks['lat'] = 90
        if 'lon' in dims:
            var_chunks['lon'] = 144
            
        encoding[var] = {'chunks': var_chunks}
    
    print("Grid variables encoding:", encoding)
    
    # Write grid information to zarr
    grid_vars.to_zarr(
        zarr_path,
        mode='a',  # Append to existing store
        # encoding=encoding,
        consolidated=True,
        compute=True
    )
    
    # Close dataset
    ds_2d.close()
    print("Added grid information variables: axyp, landfr")


if __name__ == '__main__':
    # Path configuration
    netcdf_root = '/home/serfani/serfani_data1/E3OMA2020D/'
    zarr_path = '/home/serfani/serfani_data0/E3OMA2020D.zarr'
    
    # Convert data if zarr store doesn't exist
    if not os.path.exists(zarr_path):
        print("Converting NetCDF to Zarr format (this will take some time)...")
        convert_e33oma_to_zarr(netcdf_root, zarr_path, use_dask=True, n_workers=6)



