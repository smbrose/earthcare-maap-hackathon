from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader


def _prediction_to_target_shape(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Adapts the output of the model to the shape of y.

    Dataset:
        x: [B, C_in, height, along_track]
        y: [B, C_out, along_track]

    UNet:
        pred: [B, C_out, height, along_track]

    If the model still outputs a 2D map we collapse the height dimension.
    """
    if pred.shape == y.shape:
        return pred

    if pred.ndim == 4 and y.ndim == 3:
        pred = F.adaptive_avg_pool2d(pred, output_size=(1, y.shape[-1]))
        return pred.squeeze(2)

    raise ValueError(f"Output shape {tuple(pred.shape)} incompatible with target {tuple(y.shape)}")


def _run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: torch.nn.Module,
    device: str,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            pred = _prediction_to_target_shape(pred, y)
            loss = criterion(pred, y)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    model: torch.nn.Module,
    datamodule=None,
    train_dataloader: DataLoader | None = None,
    val_dataloader: DataLoader | None = None,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str | None = None,
    criterion: torch.nn.Module | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Training loop compatible with EarthCARELightningDataModule/EarthCARELightningDataset.

    Uses batch dict:
        batch["x"] -> [B, C_in, height, along_track]
        batch["y"] -> [B, C_out, along_track]
    """
    if datamodule is not None:
        datamodule.setup("fit")
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()

    if train_dataloader is None:
        raise ValueError("Pass datamodule or train_dataloader.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = criterion or torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_dataloader, optimizer, criterion, device)
        history["train_loss"].append(train_loss)

        if val_dataloader is not None:
            with torch.no_grad():
                val_loss = _run_epoch(model, val_dataloader, None, criterion, device)
            history["val_loss"].append(val_loss)
        else:
            val_loss = None

        if verbose:
            if val_loss is None:
                logging.info("Epoch %d/%d | train_loss: %.6f", epoch + 1, epochs, train_loss)
            else:
                logging.info(
                    "Epoch %d/%d | train_loss: %.6f | val_loss: %.6f",
                    epoch + 1,
                    epochs,
                    train_loss,
                    val_loss,
                )

    return history

### Example ###
# from datamodule import EarthCARELightningDataModule
# from models.unet import UNet
# from train import train

# dm = EarthCARELightningDataModule(
#     data_dir="/path/to/patches",
#     input_vars=["var1", "var2"],
#     target_vars=["lightning_target"],
#     batch_size=8,
# )

# model = UNet(in_channels=len(dm.input_vars), out_channels=len(dm.target_vars))

# history = train(
#     model=model,
#     datamodule=dm,
#     epochs=20,
#     lr=1e-4,
# )