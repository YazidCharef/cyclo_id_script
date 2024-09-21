import xarray as xr
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import cartopy.feature as cfeature
from shapely.geometry import Point

def load_and_preprocess_data(netcdf_file, typhoon_file, sst_file, batch_size=32, start_year=None, end_year=None, use_sst=True):
    print("Starting data loading and preprocessing...")
    
    # Load typhoon data
    typhoon_data = pd.read_csv(typhoon_file, parse_dates=['Cyclogenesis Start', 'Cyclogenesis End', 
                                                          'Typhoon Start', 'Typhoon End',
                                                          'Cyclolysis Start', 'Cyclolysis End'])
    print(f"Typhoon data loaded. Total records: {len(typhoon_data)}")
    
    # Create land mask
    land_mask = create_land_mask(sst_file)
    
    # Load data
    ds_meteo = xr.open_dataset(netcdf_file)
    ds_sst = xr.open_dataset(sst_file)
    
    # Rename 'valid_time' to 'time' in SST dataset for consistency
    ds_sst = ds_sst.rename({'valid_time': 'time'})
    
    # Filter data based on years if specified
    if start_year is not None and end_year is not None:
        ds_meteo = ds_meteo.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        ds_sst = ds_sst.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        print(f"Data filtered for years {start_year} to {end_year}")
    
    # Ensure that meteo and SST datasets have the same time coordinates
    common_times = np.intersect1d(ds_meteo.time.values, ds_sst.time.values)
    ds_meteo = ds_meteo.sel(time=common_times)
    ds_sst = ds_sst.sel(time=common_times)
    
    # Create labels
    labels = create_labels(ds_meteo.time, typhoon_data)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(ds_meteo, ds_sst, land_mask, labels, batch_size, use_sst)
    
    return train_loader, val_loader, test_loader

def create_land_mask(sst_file):
    print("Creating land mask...")
    with xr.open_dataset(sst_file) as ds:
        lons, lats = np.meshgrid(ds.longitude, ds.latitude)
        land_mask = np.full(lons.shape, 0)
        land_geom = cfeature.LAND.geometries()
        for geom in land_geom:
            for i in range(land_mask.shape[0]):
                for j in range(land_mask.shape[1]):
                    point = Point(lons[i, j], lats[i, j])
                    if point.within(geom):
                        land_mask[i, j] = 1
    
    print("Land mask created.")
    return land_mask


def create_labels(time_index, typhoon_data):
    labels = np.zeros(len(time_index), dtype=int)
    for _, typhoon in typhoon_data.iterrows():
        cyclogenesis_mask = (time_index >= typhoon['Cyclogenesis Start']) & (time_index <= typhoon['Cyclogenesis End'])
        full_typhoon_mask = (time_index > typhoon['Typhoon Start']) & (time_index <= typhoon['Typhoon End'])
        cyclolysis_mask = (time_index > typhoon['Cyclolysis Start']) & (time_index <= typhoon['Cyclolysis End'])
        
        labels[cyclogenesis_mask] = 1  # Cyclogenesis
        labels[full_typhoon_mask] = 2  # Full typhoon
        labels[cyclolysis_mask] = 3  # Cyclolysis
    
    label_counts = np.bincount(labels)
    total_samples = len(labels)
    
    print("Label distribution:")
    class_names = ["No Cyclone", "Cyclogenesis", "Full Typhoon", "Cyclolysis"]
    for i, (name, count) in enumerate(zip(class_names, label_counts)):
        percentage = (count / total_samples) * 100
        print(f"{name}: {count} ({percentage:.2f}%)")
    
    return labels

class TyphoonDataset(torch.utils.data.Dataset):
    def __init__(self, ds_meteo, ds_sst, land_mask, labels, use_sst=True):
        self.ds_meteo = ds_meteo
        self.ds_sst = ds_sst
        self.land_mask = land_mask
        self.labels = labels
        self.use_sst = use_sst

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        u = self.ds_meteo['u'].isel(time=idx).values
        v = self.ds_meteo['v'].isel(time=idx).values
        r = self.ds_meteo['r'].isel(time=idx).values
        vo = self.ds_meteo['vo'].isel(time=idx).values
        
        if self.use_sst:
            sst = np.where(self.land_mask, np.nan, self.ds_sst['sst'].isel(time=idx).values)
            features = np.stack([u, v, r, vo, sst], axis=0)
        else:
            features = np.stack([u, v, r, vo], axis=0)
        
        # Normalize meteorological variables
        for i in range(4):  # u, v, r, vo
            mean = np.nanmean(features[i])
            std = np.nanstd(features[i])
            features[i] = (features[i] - mean) / (std + 1e-8)
        
        # Normalize SST separately if used
        if self.use_sst:
            sst_mean = np.nanmean(features[4])
            sst_std = np.nanstd(features[4])
            features[4] = (features[4] - sst_mean) / (sst_std + 1e-8)
        
        features = np.nan_to_num(features, nan=0.0)
        
        return torch.FloatTensor(features), torch.LongTensor([self.labels[idx]]).squeeze()

def create_data_loaders(ds_meteo, ds_sst, land_mask, labels, batch_size, use_sst):
    print("Creating data loaders...")
    dataset = TyphoonDataset(ds_meteo, ds_sst, land_mask, labels, use_sst)
    
    total_size = len(dataset)
    test_size = int(0.2 * total_size)
    val_size = int(0.2 * total_size)
    train_size = total_size - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"DataLoaders created. Sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")
    return train_loader, val_loader, test_loader
