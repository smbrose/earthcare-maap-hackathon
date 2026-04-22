from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import lightning as L
import os
from pathlib import Path
import random
import xarray as xr
import numpy as np
from dataset import EarthCARELightningDataset


def compute_input_stats(filepaths, input_vars, output_path=None):
    output_dict = {
        var: {
            "sum": 0.0,
            "counts": 0,
            "sq_sum": 0.0,
        }
        for var in input_vars
    }

    for file in filepaths:
        ds = read_one_patch(file)
        try:
            for var in input_vars:
                arr = ds[var].values.astype(np.float32)
                mask = np.isfinite(arr)
                valids = arr[mask]

                if valids.size == 0:
                    continue

                output_dict[var]["sum"] += float(valids.sum())
                output_dict[var]["counts"] += int(valids.size)
                output_dict[var]["sq_sum"] += float(np.square(valids).sum())

        finally:
            if hasattr(ds, "close"):
                ds.close()

    for values in output_dict.values():
        values["mean"] = values["sum"] / max(values["counts"], 1)
        var = values["sq_sum"] / max(values["counts"], 1) - values["mean"] ** 2
        values["std"] = float(np.sqrt(max(var, 1e-6)))

        values["sum"] = float(values["sum"])
        values["sq_sum"] = float(values["sq_sum"])
        values["counts"] = int(values["counts"])
        values["mean"] = float(values["mean"])

    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(output_dict, f, indent=2)

    return output_dict   

def make_filelist(dataset_path):
    dataset_path = Path(dataset_path)
    return sorted(str(f) for f in dataset_path.iterdir() if f.suffix == ".h5")

def read_one_patch(file):
    return xr.open_dataset(file)
    
def random_split_dataset(
    dataset_dir,
    train_ratio=0.7,
    val_ratio=0.20,
    test_ratio=0.10,
    seed=42):

    dataset_dir = Path(dataset_dir)

    files = make_filelist(dataset_dir)

    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    split_dict = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    return split_dict


class EarthCARELightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        input_vars,
        target_vars,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        fill_value: float = 0.0,
        norm_with_train: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.splits_dict = random_split_dataset(data_dir)
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.fill_value = fill_value
        self.norm_with_train = norm_with_train           
            

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_files = self.splits_dict["train"]
        val_files = self.splits_dict["val"]
        test_files = self.splits_dict["test"]

        if self.norm_with_train:
            mean_std_dict_train = compute_input_stats(train_files, self.input_vars)
            mean_std_dict_val = mean_std_dict_train
            mean_std_dict_test = mean_std_dict_train
        else:
            mean_std_dict_train = compute_input_stats(train_files, self.input_vars)
            mean_std_dict_val = compute_input_stats(val_files, self.input_vars)
            mean_std_dict_test = compute_input_stats(test_files, self.input_vars)

        self.train_dataset = EarthCARELightningDataset(
            filelist=train_files,
            input_vars=self.input_vars,
            target_vars=self.target_vars,
            mean_std_dict=mean_std_dict_train,
        )

        self.val_dataset = EarthCARELightningDataset(
            filelist=val_files,
            input_vars=self.input_vars,
            target_vars=self.target_vars,
            mean_std_dict=mean_std_dict_val,
            fill_value=self.fill_value,
        )

        self.test_dataset = EarthCARELightningDataset(
            filelist=test_files,
            input_vars=self.input_vars,
            target_vars=self.target_vars,
            mean_std_dict=mean_std_dict_test,
            fill_value=self.fill_value,
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )