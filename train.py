"""Train, validate, and checkpoint the best lunar terrain classifier."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

import config
from dataset import LunarTerrainDataset
from model import create_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, optimizer, device, training: bool, scaler: GradScaler | None = None,
              accumulation_steps: int = 1, use_amp: bool = False):
    model.train(training)
    losses, actual, probabilities = [], [], []
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        azimuth = batch["azimuth"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training), autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, azimuth)
            loss = criterion(logits, labels) / accumulation_steps

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item() * accumulation_steps)
        actual.extend(labels.cpu().numpy())
        probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy())

    if training and (len(loader) % accumulation_steps != 0):
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return float(np.mean(losses)), balanced_accuracy_score(actual, predictions), np.asarray(probabilities)


def best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Select a validation threshold for balanced accuracy without fitting on test data."""
    candidates = np.unique(np.concatenate(([0.5], probabilities)))
    scores = [balanced_accuracy_score(labels, probabilities >= threshold) for threshold in candidates]
    index = int(np.argmax(scores))
    return float(candidates[index]), float(scores[index])


def main() -> None:
    seed_everything(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    metadata = pd.read_csv(config.TRAIN_METADATA)
    train_df, valid_df = train_test_split(metadata, test_size=config.VALIDATION_SIZE,
                                           stratify=metadata["label"], random_state=config.RANDOM_SEED)

    train_ds = LunarTerrainDataset(train_df, config.TRAIN_IMAGE_DIR, training=True)
    valid_ds = LunarTerrainDataset(valid_df, config.TRAIN_IMAGE_DIR, training=False)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=pin_memory,
        persistent_workers=config.PERSISTENT_WORKERS and config.NUM_WORKERS > 0,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=pin_memory,
        persistent_workers=config.PERSISTENT_WORKERS and config.NUM_WORKERS > 0,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
    )

    model = create_model(
        pretrained=True,
        metadata_dropout=config.METADATA_DROPOUT,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
        channels_last=config.USE_CHANNELS_LAST,
        compile_model=config.COMPILE_MODEL,
    ).to(device)

    counts = np.bincount(train_df["label"].astype(int), minlength=2)
    pos_weight = torch.tensor([counts[0] / max(counts[1], 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    scaler = GradScaler(device.type, enabled=config.USE_AMP) if device.type == "cuda" else None

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_score = -np.inf

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_bacc, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, True,
            scaler, config.GRADIENT_ACCUMULATION_STEPS, config.USE_AMP
        )
        valid_loss, _, valid_probs = run_epoch(
            model, valid_loader, criterion, optimizer, device, False,
            scaler, 1, config.USE_AMP
        )
        scheduler.step()
        valid_labels = valid_df["label"].to_numpy(dtype=int)
        threshold, valid_bacc = best_threshold(valid_labels, valid_probs)
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} bacc {train_bacc:.4f} | "
              f"valid loss {valid_loss:.4f} bacc {valid_bacc:.4f} threshold {threshold:.3f}")
        if valid_bacc > best_score:
            best_score = valid_bacc
            torch.save({
                "model_state": model.state_dict(),
                "best_bacc": best_score,
                "epoch": epoch,
                "threshold": threshold,
                "config": {
                    "batch_size": config.BATCH_SIZE,
                    "lr": config.LEARNING_RATE,
                    "weight_decay": config.WEIGHT_DECAY,
                    "metadata_dropout": config.METADATA_DROPOUT,
                    "classifier_dropout": config.CLASSIFIER_DROPOUT,
                    "image_size": config.IMAGE_SIZE,
                }
            }, config.BEST_CHECKPOINT)
    print(f"Best validation balanced accuracy: {best_score:.4f}; checkpoint: {config.BEST_CHECKPOINT}")


if __name__ == "__main__":
    main()
