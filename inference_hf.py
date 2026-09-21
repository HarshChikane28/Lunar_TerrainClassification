"""Run inference using model downloaded from Hugging Face Hub."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

import config
from dataset import LunarTerrainDataset
from model import create_model


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download checkpoint from HF Hub
    print("Downloading model from Hugging Face Hub...")
    checkpoint_path = hf_hub_download(
        repo_id="harsh-101/Lunar",
        filename="best_model.pt"
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Setup data
    metadata = pd.read_csv(config.TEST_METADATA)
    dataset = LunarTerrainDataset(metadata, config.TEST_IMAGE_DIR, training=False)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=config.PERSISTENT_WORKERS and config.NUM_WORKERS > 0
    )

    # Create model and load weights
    model = create_model(
        pretrained=False,
        metadata_dropout=config.METADATA_DROPOUT,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
        channels_last=config.USE_CHANNELS_LAST,
        compile_model=config.COMPILE_MODEL,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    threshold = float(checkpoint.get("threshold", 0.5))
    predictions = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        azimuth = batch["azimuth"].to(device, non_blocking=True)
        logits = model(images, azimuth)
        predictions.extend((torch.sigmoid(logits).cpu().numpy() >= threshold).astype(int))

    submission = pd.DataFrame({"image_id": metadata["image_id"], "label": np.asarray(predictions, dtype=int)})
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Wrote {len(submission)} predictions to {config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()