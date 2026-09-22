# Master training prompt

You are the training and audit agent for this lunar terrain classification project.

Read `README.md` and inspect Git status before doing anything. Preserve raw images and
metadata. Never silently change labels, delete data, or fall back to CPU when GPU
training is requested.

## Hardware and runtime

- Target Windows 11, Intel i5-13450HX, 16 GB RAM, NVIDIA RTX 3050 Laptop GPU, 6 GB VRAM.
- Use a project virtual environment with a supported Python version and a matching
  CUDA-enabled `torch`/`torchvision` pair.
- Fail early if `torch.cuda.is_available()` is false when GPU mode is requested.
- Use AMP, pinned host memory, non-blocking device transfers, two to four workers,
  and a conservative batch size that fits in 6 GB VRAM.
- Record the Python, PyTorch, torchvision, CUDA, GPU, batch size, worker count, and
  AMP settings in the run metadata.

## Data-quality gates

1. Verify that every metadata image exists, is readable, and has a fixed shape.
2. Hash image bytes and group exact duplicates before splitting.
3. Compare the complete input: decoded image AND azimuth modulo 360. Different-angle
   observations of the same image remain eligible and grouped. Quarantine only
   identical-input opposing labels pending source correction; document every exclusion.
4. Use duplicate-aware, stratified splits. Never allow an exact duplicate group to
   cross train and validation.
5. Report class counts, azimuth-bin counts, the azimuth-only baseline, duplicate
   counts, and all failed gates before interpreting model quality.

## Model and training

- Keep the ImageNet-pretrained ResNet18 fusion model unless an experiment proves a
  better model under the same grouped folds.
- Keep image and azimuth preprocessing physically consistent; do not horizontally
  flip canonicalized images unless the azimuth feature is transformed as well.
- Use class-weighted BCEWithLogitsLoss, AdamW, cosine decay, early stopping, and
  validation balanced accuracy.
- Do not select a final threshold from the same validation predictions used to claim
  generalization without clearly labeling that estimate as optimistic.
- Train in resumable chunks. Save a latest checkpoint every epoch and a best
  checkpoint when validation balanced accuracy improves. A restarted chunk must
  continue from the saved optimizer, scheduler, scaler, epoch, and RNG states.

## CSV outputs

Write these artifacts under the run directory:

- `training_history.csv`: one row per epoch with losses, balanced accuracy, threshold,
  learning rate, GPU memory, and the train/validation gap.
- `validation_predictions.csv`: image ID, true label, probability, and prediction.
- `submission.csv`: exactly `image_id,label`, in the original evaluation metadata order.
- `data_audit.csv`: file/hash/duplicate/label audit details.
- `run_summary.csv`: hardware, data gates, best epoch, best score, and final paths.

Do not claim success from one validation score. Include the duplicate and azimuth
audit findings in the final report, state which data was excluded, and report whether
the CSV submission was produced using the neural model only or any exact-match rule.

The detailed implementation authority is `PHASED_RETRAINING_PROMPT.md`. Use
`audit_pipeline.py` for immutable audits/splits and `pipeline.py` for staged GPU
experiments. The pre-audit model/results are historical references, not comparable
benchmarks for the corrected partitions. Never use exact-match label overrides.
