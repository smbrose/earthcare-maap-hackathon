import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import random
import json
import os

def read_one_patch(file):
    return xr.open_dataset(file)

# def load_input_stats(json_path):
#     with open(json_path, "r", encoding="utf-8") as f:
#         stats = json.load(f)
#     return stats


class EarthCARELightningDataset(Dataset):
    def __init__(
        self,
        filelist,
        input_vars,
        target_vars,
        mean_std_dict=None,
        fill_value = 0.0
    ):

        self.filelist = filelist
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.stats_dict = mean_std_dict
        self.fill_value = fill_value

    def __len__(self):
        return len(self.filelist)

    def _prepare_input_array(self,
                             ds,
                             var):
        da = ds[var]
        arr = da.transpose("height", "along_track").values.astype(np.float32)

        arr = np.nan_to_num(arr, nan=self.fill_value)
        arr = (arr - self.stats_dict[var]["mean"]) / self.stats_dict[var]["std"]

        return arr

    def _prepare_target_array(self,
                               ds, 
                               var):
        da = ds[var]

        arr = da.transpose("along_track").values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=self.fill_value)

        return arr


        


    def __getitem__(self, ind):
        file_path = self.filelist[ind]
        ds = read_one_patch(file_path)

        try:
            x_channels = []
            y_channels = []
            for var in self.input_vars:
                x_arr = self._prepare_input_array(ds, var)
                x_channels.append(x_arr)

            x = np.stack(x_channels, axis=0)

            for var in self.target_vars:
                y_arr = self._prepare_target_array(ds,var)
                y_channels.append(y_arr)

            y = np.stack(y_channels, axis=0)

            sample = {
                "x": torch.from_numpy(x).float(),
                "y": torch.from_numpy(y).float(),
                "path": str(file_path),
            }

            return sample

        except Exception as e:
            print(e)