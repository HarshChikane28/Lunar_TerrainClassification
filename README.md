---
library_name: pytorch
pipeline_tag: image-classification
tags:
- computer-vision
- image-classification
- lunar-terrain
---

# The Pareidolia Paradox — Lunar Terrain Classification

Binary image classification of lunar terrain: `0 = Depth`, `1 = Rise`. Each observation contains a 256×256 grayscale image and its acquisition `sun_azimuth_angle`.

## 🏆 Trained Model & Accuracy

| Metric | Result |
|---|---|
| **Final trained model** | **ResNet18 `raw_fusion` — image + sun-azimuth fusion** |
| Task | Lunar Terrain Classification |
| Development performance | **76.67% mean balanced accuracy** across five grouped folds |
| Untouched holdout performance | **76.40% balanced accuracy** on 1,122 observations |
| Final training data | 7,852 eligible labeled observations |
| Local final checkpoint | [`inference.pt`](</C:/Users/athar/OneDrive/Desktop/Lunar_TerrainClassification/runs/phasewise_v3/models/final_refit/inference.pt>) |
| Submitted CSV | [`submission.csv`](</C:/Users/athar/OneDrive/Desktop/Lunar_TerrainClassification/submission.csv>) |

The final model uses the unrotated image together with the original sun angle. The frozen decision threshold is `0.48025015`. The holdout score is balanced accuracy, not ordinary accuracy. The simple fixed-angle baseline scored 78.29% on the same holdout, so the neural model should not be described as superior to that baseline.

## Methodology Summary

### Data preparation

- Images are validated as 256×256 8-bit grayscale-equivalent images and normalized with ImageNet statistics after replication to three channels.
- Mild brightness/contrast augmentation is applied during training; no random rotations or flips are used.
- Of 7,854 training rows, 7,852 were eligible. Two identical-image, identical-angle rows with conflicting labels were quarantined without relabeling.
- Different-angle observations of the same image were retained and kept in the same group.
- The evaluation uses five grouped development folds plus an untouched 1,122-row holdout. The final fit uses all 7,852 eligible rows.

### Model architecture

- Image branch: ImageNet-pretrained torchvision ResNet18.
- Metadata branch: `sun_azimuth_angle` encoded as `[sin(θ), cos(θ)]`, then processed by an MLP `2 → 32 → 16`.
- The image and metadata features are concatenated and passed through a `528 → 128 → 1` classifier for binary prediction.
- The production variant is `raw_fusion`: image pixels are not rotated, and the original acquisition angle is supplied to the metadata branch.

### Sun-azimuth rotation handling

Canonical rotation was tested as a controlled alternative. Images were rotated about their center with OpenCV using both `-sun_azimuth_angle` and the opposite sign. The canonicalized variants used a 256×256 canvas and neutral-gray padding for exposed borders. The original acquisition angle remained available as metadata.

The dataset’s physical angle convention is undocumented, so rotation was not assumed to be correct. Five-fold results selected unrotated fusion: `raw_fusion` achieved 76.67% mean balanced accuracy versus 76.09% for canonicalized fusion. Therefore, the final model does not rotate pixels; it uses the image plus sine/cosine angle encoding.

### Training and evaluation

- Loss: weighted `BCEWithLogitsLoss`, with class weighting computed from each training partition.
- Optimizer: AdamW.
- Learning rates: `3e-5` for the ResNet backbone and `3e-4` for the fusion/classifier head.
- Weight decay: `1e-4`; classifier dropout: `0.3`.
- Batch size: `32`; CUDA AMP enabled; two persistent data-loader workers.
- Scheduler: cosine annealing with a two-epoch backbone warm-up.
- Development runs used an eight-epoch cap with patience four early stopping. The selected final refit used a fixed four-epoch schedule on all eligible rows.
- Checkpoint selection used grouped development selection data. The decision threshold was then calibrated from development out-of-fold predictions and frozen before holdout evaluation.
- Metrics include balanced accuracy, class recall, ROC-AUC, log loss, confusion counts, and group-bootstrap uncertainty.

The local audited final checkpoint is `runs/phasewise_v3/models/final_refit/inference.pt`. The legacy `runs/resnet18_fusion/best_model.pt` belongs to the earlier experiment and is not the final phasewise_v3 production checkpoint.

## 🤗 Hugging Face Model

The model artifact is available on Hugging Face for convenient access and independent evaluation:

**Hugging Face:** [Crystalbullet/lunar_project](https://huggingface.co/Crystalbullet/lunar_project)

Relevant hosted model file: `best_model.pt`.

## 🔗 Project Resources

- **🤗 Hugging Face Model:** [Crystalbullet/lunar_project](https://huggingface.co/Crystalbullet/lunar_project)
- **📄 CSV File:** The exact Google Drive CSV URL is not present in the repository or available project context. The verified local submission is [`submission.csv`](</C:/Users/athar/OneDrive/Desktop/Lunar_TerrainClassification/submission.csv>).
- **📊 Full evaluation report:** [`IMPLEMENTATION_REPORT.md`](</C:/Users/athar/OneDrive/Desktop/Lunar_TerrainClassification/runs/phasewise_v3/IMPLEMENTATION_REPORT.md>)

## Project Overview

The project investigates whether lunar terrain appearance can be classified reliably when illumination direction is provided as metadata. It compares image-only, canonicalized-image, angle-only, fixed-angle-rule, and image-plus-angle models using grouped evaluation and a sealed holdout.

## Features

- Audited image, metadata, duplicate, and train/evaluation-overlap checks.
- Grouped nested cross-validation with a sealed holdout.
- GPU training with AMP, resumable epoch checkpoints, and strict configuration fingerprints.
- Tested azimuth canonicalization and opposite rotation conventions.
- Threshold calibration from development out-of-fold predictions.
- CSV schema and checkpoint-reload validation for all 2,000 evaluation rows.
- Grad-CAM and illumination-shift diagnostics.

## Project Structure

```text
Train_DATA/                         Training images and metadata
Test_DATA/                          Evaluation images and metadata
dataset.py                          Image loading, normalization, augmentation, angle encoding
model.py                            ResNet18 image/metadata fusion model
training_engine.py                  GPU trainer, checkpointing, early stopping, evaluation
pipeline.py                         Audited benchmark, comparison, tuning, holdout, and refit pipeline
predict.py                          Checkpoint-based inference and CSV validation
test_pipeline.py                    Contract and GPU resume tests
runs/phasewise_v3/                  Final experiment artifacts and reports
submission.csv                      Validated 2,000-row prediction file
```

## Installation / Setup

The verified environment uses Python 3.12, PyTorch `2.11.0+cu128`, torchvision `0.26.0+cu128`, and an NVIDIA GPU. The project requires CUDA for production training and inference.

```powershell
.venv\Scripts\python.exe -m unittest test_pipeline
.venv\Scripts\python.exe -u pipeline.py --stage all
```

For a fresh environment, install the CUDA PyTorch wheel pair from the official CUDA 12.8 index before installing `requirements.txt`. Use only `opencv-python-headless`, not multiple OpenCV packages providing the same `cv2` namespace.

## Usage / Inference

```powershell
.venv\Scripts\python.exe predict.py `
  --checkpoint runs/phasewise_v3/models/final_refit/inference.pt `
  --output new_submission.csv
```

Inference validates the exact columns `image_id,label`, evaluation-row order, IDs, null handling, and binary integer labels. Probabilities are written beside the requested output as `evaluation_probabilities.csv`.

## Training

Run the complete restartable workflow:

```powershell
.venv\Scripts\python.exe -u pipeline.py --stage all
```

Or run one explicit fold in chunks:

```powershell
.venv\Scripts\python.exe train.py `
  --run-id example_f0 `
  --variant raw_fusion `
  --fold 0 `
  --total-epochs 8 `
  --chunk-epochs 5
```

Checkpoints contain model, optimizer, scheduler, AMP scaler, RNG state, configuration, data fingerprints, and training history. A resumed run must use a compatible checkpoint and configuration.

## Evaluation / Results

| Evaluation | Balanced accuracy |
|---|---:|
| Five-fold development mean — raw fusion | **76.67%** |
| Five-fold development mean — canonicalized fusion | 76.09% |
| Untouched holdout — selected neural model | **76.40%** |
| Untouched holdout — fixed-angle rule | **78.29%** |
| 270°–315° illumination-shift diagnostic | 50.00% |

The 50.00% shift diagnostic indicates weak generalization to that held-out lighting range. The holdout was evaluated before the all-data refit; the refitted checkpoint has no independent labeled test evaluation afterward.

## Future Improvements

- Verify the physical meaning and coordinate convention of `sun_azimuth_angle`.
- Improve robustness to unseen illumination ranges and evaluate on an independently collected test set.
- Investigate scene-boundary shortcuts identified by Grad-CAM.
- Compare calibrated ensembles or stronger backbones only with additional grouped validation evidence.

## License / Credits

No license file is currently included. The model and experiment artifacts are provided for evaluation and hackathon use; add a project-specific license before redistribution.
