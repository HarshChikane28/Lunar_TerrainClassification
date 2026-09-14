"""Dataset and preprocessing for canonicalized lunar images."""

from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

import config


def encode_azimuth(angle_degrees: float) -> torch.Tensor:
    """Encode circular azimuth as sin/cos, avoiding a discontinuity at 0/360."""
    radians = np.deg2rad(float(angle_degrees))
    return torch.tensor([np.sin(radians), np.cos(radians)], dtype=torch.float32)


def rotate_to_canonical(image: np.ndarray, azimuth_degrees: float,
                        canonical_degrees: float = config.CANONICAL_AZIMUTH_DEGREES) -> np.ndarray:
    """Rotate the image so the recorded sun direction points to one direction.

    We use OpenCV's image-plane convention: positive angles in
    ``getRotationMatrix2D`` are counter-clockwise on the displayed image.
    Therefore an image whose sun azimuth is ``a`` is rotated by ``canonical-a``.
    This is the key topographic-inversion correction: the same physical feature
    is presented with the same lighting direction regardless of acquisition.
    If the source dataset defines azimuth in the opposite image-plane direction,
    change the sign in the one line below and keep train/inference identical.
    """
    rotation_degrees = float(canonical_degrees) - float(azimuth_degrees)
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def build_transforms(training: bool) -> A.Compose:
    """Only augment after canonical rotation; no vertical flip/random rotation."""
    operations = []
    if training:
        operations.extend([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        ])
    operations.extend([
        A.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return A.Compose(operations)


class LunarTerrainDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame | str | Path, image_dir: str | Path,
                 training: bool = False):
        self.metadata = pd.read_csv(metadata) if isinstance(metadata, (str, Path)) else metadata.reset_index(drop=True)
        required = {"image_id", "sun_azimuth_angle"}
        missing = required - set(self.metadata.columns)
        if missing:
            raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
        self.has_labels = "label" in self.metadata.columns
        if training and not self.has_labels:
            raise ValueError("Training dataset requires a label column.")
        self.image_dir = Path(image_dir)
        self.transforms = build_transforms(training)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image_path = self.image_dir / str(row["image_id"])
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = rotate_to_canonical(image, row["sun_azimuth_angle"])
        transformed = self.transforms(image=image)
        item = {
            "image": transformed["image"],
            "azimuth": encode_azimuth(row["sun_azimuth_angle"]),
            "image_id": str(row["image_id"]),
        }
        if self.has_labels:
            item["label"] = torch.tensor(float(row["label"]), dtype=torch.float32)
        return item
