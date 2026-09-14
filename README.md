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

## Human-style training and audit workflow

Do not treat a near-100% train or validation score as automatic success. On an
ambiguous terrain task, an unusually high score is first a reason to investigate
leakage and shortcuts. Always report the train/validation gap alongside the
validation balanced accuracy, and never remove dropout or reduce augmentation
only to increase training accuracy.

Use the following workflow before accepting a model:

1. **Verify the data split.** Keep the split stratified by label and check that
   no duplicate or near-duplicate images occur across train and validation. If
   the dataset contains groups, acquisition sessions, or related image IDs,
   split by group rather than allowing related samples in both partitions.
2. **Check metadata shortcuts before training.** Summarize label counts by
   azimuth bins and measure how well `sun_azimuth_angle` alone predicts the
   label. A strong azimuth-only result is suspicious and must be explained
   before trusting an image model.
3. **Use stratified k-fold cross-validation.** Report every fold's balanced
   accuracy, the mean, and the standard deviation. Do not accept a result based
   on one lucky train/validation split. Keep a final untouched holdout for the
   last check when possible.
4. **Watch the gap.** Train accuracy near 100% with validation also near 100%
   can be legitimate, but train near 100% with noticeably lower validation
   performance is overfitting regardless of how good the validation number
   looks in isolation. Investigate suspicious validation scores above roughly
   90--95% for leakage or shortcuts rather than immediately shipping them.
5. **Run an ablation.** Train one model with the azimuth branch and one without
   it, using the same folds and preprocessing. A meaningful drop without azimuth
   supports its value. No drop means it may be unnecessary; an unexpectedly
   strong azimuth-only model means the data or split needs investigation.
6. **Inspect model attention.** Generate Grad-CAM images for correct and
   incorrect predictions from every class. The activation should focus on lunar
   terrain and physically meaningful rim/shadow regions, not image corners,
   borders, compression marks, or other artifacts.
7. **Inspect predictions manually.** Select at least 20 random evaluation or
   validation predictions, record the confidence and predicted class, and view
   the images. Include difficult, high-confidence, and incorrect examples in
   the experiment record.

### Agentic acceptance protocol

An automated training agent should work in stages and stop for investigation
when a gate fails:

```text
audit metadata and file identities
    -> audit azimuth-only baseline
    -> create stratified grouped/k-fold splits
    -> train image+azimuth and image-only ablations
    -> report train score, validation score, gap, mean, and standard deviation
    -> run leakage checks and Grad-CAM review
    -> sample 20 predictions for human inspection
    -> accept, revise, or stop with an audit report
```

The agent must not claim success from the validation score alone. Its report
should include:

- per-fold train and validation balanced accuracy, plus their gap;
- mean and standard deviation across folds;
- label counts and azimuth-bin summaries;
- the azimuth-only baseline and image-only ablation;
- duplicate/near-duplicate split-check results;
- paths to Grad-CAM visualizations and the 20-image prediction sample;
- any failed gate, suspected shortcut, or unresolved data-quality issue.

The current `train.py` provides the baseline stratified split, weighted loss,
validation balanced accuracy, threshold selection, and best-checkpoint saving.
The k-fold runner, duplicate detector, azimuth-only baseline, Grad-CAM report,
and prediction gallery should be added as separate audit utilities before
claiming that the full protocol has been completed. This keeps the workflow
reproducible and prevents an agent from quietly changing the experiment until a
metric improves.
