# The Pareidolia Paradox

Binary lunar terrain classification: class `0` is Depth (craters/depressions),
and class `1` is Rise (mounds/rocks).

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Extract the supplied archives so the layout is:

```text
Train_DATA/
  train_metadata.csv
  train_images/train_00001.png ...
Test_DATA/
  test_metadata.csv
  eval_images/eval_00001.png ...
```

The default paths are in `config.py`; change them there if your folders differ.

## Design

Each image is rotated by `canonical_azimuth - sun_azimuth_angle` before any
augmentation. This puts illumination into a common image-plane direction and
reduces the crater/mound shadow ambiguity. Azimuth is also passed to the model
as `(sin(angle), cos(angle))`, preserving its circular nature. Only horizontal
flip and brightness/contrast jitter are used after rotation; there is no vertical
flip or random rotation.

The image branch is ImageNet-pretrained ResNet18 with its first convolution
converted to grayscale by averaging the pretrained RGB filters. A small
`2 -> 32 -> 16` MLP encodes azimuth and is fused with the 512-dimensional image
feature before binary classification.

## Train and predict

These commands perform the actual computation only when you run them manually:

```bash
python train.py
python predict.py
```

`train.py` makes a stratified split, uses class-weighted
`BCEWithLogitsLoss`, AdamW, cosine decay, and saves the best validation balanced
accuracy checkpoint to `checkpoints/best_model.pt`. On every validation pass it
selects the threshold that maximizes balanced accuracy and stores that threshold
in the checkpoint. `predict.py` loads the checkpoint, performs deterministic
preprocessing, and writes exactly `image_id,label` to `submission.csv`.

No training or inference is run as part of creating this project.
