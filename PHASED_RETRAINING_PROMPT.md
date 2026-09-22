# Phased implementation prompt: Pareidolia Paradox

Implement the following phases in this repository, verify each phase, and then retrain on the NVIDIA GPU in resumable chunks. The objective is reliable held-out balanced accuracy and a valid prediction CSV. Do not promise an accuracy target. Follow the supplied challenge description: the input is the joint observation (grayscale image, sun_azimuth_angle), and the production model must use both.

Repository: C:/Users/athar/OneDrive/Desktop/Lunar_TerrainClassification

Hardware: Windows 11, Intel i5-13450HX, approximately 16 GB RAM, RTX 3050 Laptop GPU with 6 GB VRAM. The existing .venv previously verified Python 3.12.14, torch 2.11.0+cu128 and torchvision 0.26.0+cu128. Verify the current environment before changing it.

## Corrections to the previous experiment

- The previous experiment excluded 2,916 rows solely because identical image bytes had different labels. That is insufficient evidence of a label error in this illumination-conditioned task. Different sun angles can change the interpretation of the same appearance. Reassess these rows using the complete model input.
- Inspection of the saved audit found 1,458 opposite-label image pairs, but only one pair with the same recorded angle: train_02966.png and train_04190.png, both 210.02 degrees. Recompute this from source images and metadata; do not assume every other pair is physically valid merely because the angles differ.
- The 0.7835 azimuth-rule score was descriptive, on the original full training metadata. The 0.7435 neural-model score was selected and threshold-tuned on a validation subset after exclusions. They are not a controlled comparison.
- The previous run used only the first split produced by StratifiedGroupKFold, not full five-fold cross-validation.
- Previous train metrics were measured during parameter updates, with augmentation/dropout and threshold 0.5; validation used evaluation mode and an optimized threshold. The reported gap was not an apples-to-apples generalization gap.
- Early stopping was a manual decision after ten epochs, not implemented in the runner. Chunk resume omitted RNG and augmentation state.

## Phase 0 — Preserve evidence and establish run identity

Read the challenge brief, README, all project source, existing master prompt and saved run artifacts. Inspect Git status and preserve existing edits, including metadata edits. Do not reset the worktree or overwrite old checkpoints/submissions.

Create a new experiment directory and capture source revision, dirty diff, environment versions, configuration, input fingerprints and a run ID. Update outdated README/master-prompt statements as implementation changes. Use explicit new-run and resume modes; never resume the old ten-epoch experiment into a new data/model configuration.

Gate: historical artifacts are identifiable and every future output belongs to one immutable experiment configuration.

## Phase 1 — Reassess all 7,854 training observations

Validate train/test columns, unique IDs, label values, finite angles, image existence, decoded shape, datatype and corruption. Verify whether RGB files have identical grayscale channels. Compute both file SHA-256 and decoded-pixel hashes so compression differences do not hide identical images.

Audit duplicate groups using image identity AND angle modulo 360. Report the minimum circular angular difference between paired observations. Separate:

1. Same image and same angle, same label: repeated equivalent observations; keep them in one group and document deduplication or weighting.
2. Same image and different angle, different label: potentially valid illumination-conditioned examples; retain by default, keep in one split group, and investigate their geometry.
3. Same image and same recorded angle, different label: unresolved contradictory targets; quarantine the full affected input pair in a separate manifest pending authoritative correction.
4. Near-duplicates or related scenes: identify candidates using perceptual similarity; verify before combining groups. Never infer that near-duplicates must share labels.

Do not relabel samples from a model prediction or arbitrary majority vote. Keep the raw files untouched. Remove the existing blanket pixel-conflict exclusion. If the previous findings are confirmed and no other issues exist, 7,852 observations would remain eligible after quarantining the two identical-input conflicts. Report verified counts rather than hardcoding this expectation.

Audit train/evaluation image and joint-input overlap separately. Do not copy training labels onto evaluation images: different angles can imply different classes, and the brief forbids image-specific prediction rules.

Outputs: data_audit.csv, duplicate_pairs.csv, excluded_samples.csv, class_azimuth_counts.csv and overlap_audit.csv. Every exclusion needs a reason and original metadata.

Gate: every source row is accounted for; legitimate different-angle pairs remain available; unresolved contradictions are explicitly reported.

## Phase 2 — Verify illumination geometry

Isolate canonical rotation in one tested function. Define angle origin, clockwise/counterclockwise direction, image coordinates and the meaning of the incoming light direction. A canonical angle of zero does not automatically mean image-up.

Save annotated original/canonicalized image pairs spanning both classes and all angle quadrants, including duplicate-image pairs with different angles. Create synthetic direction-marker tests for wrapping, positive/negative rotations and known right-angle transformations. Distinguish testing implementation mathematics from verifying the dataset's real angle convention.

If the source convention remains undocumented, record the uncertainty and compare a small preregistered set of conventions using development data only. Include a raw-image-plus-azimuth baseline. Never choose the convention using evaluation predictions or a final holdout.

Disable geometric augmentation until its image/angle transformation is tested. Brightness/contrast augmentation can stay mild. Reflection padding can create terrain-like border artifacts: inspect borders and compare a documented crop/mask/padding alternative if needed.

Distinguish acquisition azimuth from the illumination angle after transformation. After canonicalization, the transformed direction is constant. If the original acquisition angle is retained as an auxiliary input, name it explicitly and test whether it adds a shortcut. A canonicalized image-only ablation still uses azimuth through preprocessing; it is not a truly metadata-free baseline.

Gate: train, validation and inference share one geometry implementation and all known ambiguities are recorded.

## Phase 3 — Lock splits and evaluation rules

Build groups from verified image identities and available scene/acquisition relationships. All examples of one source image, including opposite-label angle variants, must stay in the same partition.

Reserve approximately 15% of groups as a final untouched holdout, subject to adequate class coverage. Create five development folds with representative label and azimuth coverage. Use label-by-angle strata where group/sample counts permit, merge sparse bins deterministically, and report actual coverage. Assert disjoint groups in every split. Persist all sample assignments and their fingerprints.

For each outer development fold, use only its training groups for checkpoint selection and threshold calibration. Reserve inner selection/calibration groups or use a documented inner cross-validation scheme. Evaluate the outer fold after those decisions are fixed. Do not tune a threshold on the outer fold and call its resulting score unbiased.

Use the same folds and eligible rows for all baselines and ablations. Use an angle-held-out diagnostic, with shared image groups excluded from its training partition, to probe sensitivity to changes in illumination distribution.

Outputs: split_manifest.csv, split_summary.csv, split_checks.csv.

Gate: no group leakage, both classes represented, calibration/selection/evaluation roles explicit, final holdout sealed.

## Phase 4 — Repair the trainer and checkpoint format

Implement these concrete repairs before substantial training:

- Honor the actual --resume path; current code ignores its supplied value. Require compatibility of model, preprocessing, dataset and split fingerprints before resuming.
- Make --total-epochs control the scheduler horizon. Chunk length must not reset learning rates or early-stopping counters.
- Save model, optimizer, scheduler, AMP scaler, epoch, best score, best epoch, threshold, early-stopping state, resolved configuration, fingerprints and RNG states.
- Account for Python, NumPy, CPU/CUDA Torch, DataLoader and augmentation randomness. With persistent workers, saving the main-process RNG alone is insufficient. Prefer deterministic per-sample/per-epoch augmentation seeds or explicitly restorable worker state.
- Test uninterrupted versus chunk-resumed training on a tiny fixed subset. Document numerical tolerances; do not claim bitwise reproducibility with nondeterministic kernels.
- Write checkpoints atomically and make CSV recovery consistent with the committed checkpoint. Do not duplicate or omit epochs on resume.
- Wire config dropout values into the model constructor; they currently rely on constructor defaults.
- Compute sample-weighted epoch loss and metrics from sigmoid(logits.float()) to avoid quantizing threshold decisions in FP16.
- Compute train-evaluation and validation metrics in eval mode, with deterministic transforms, the same checkpoint and the same independently selected threshold. Keep augmented online train metrics separately labeled.
- Implement automatic early stopping using inner selection metrics, with configurable patience/minimum improvement. Stop before the next chunk if patience is exhausted.
- Replace repeated quadratic threshold scoring with a tested sort/cumulative-count method or bounded candidate grid. Define threshold tie-breaking and >= semantics.
- Load architecture, channel count, normalization and inference precision from checkpoint configuration. Reproduce validation predictions after checkpoint reload.
- Validate chunk sizes, positive epoch limits, class presence and finite losses. Require CUDA for both production training and inference.

Gate: dataset, group separation, geometry, model shape, thresholding, resume, checkpoint reload and submission schema tests pass.

## Phase 5 — Tune the runtime for this PC

Use the existing project environment if its CUDA test passes. Record GPU name, CUDA build and package versions, and lock the verified dependencies. Avoid installing competing OpenCV packages that provide the same cv2 namespace.

Start with AMP, batch size 16, two workers, pinned memory and non-blocking transfers. Use persistent workers/prefetch only when worker count is positive. Limit OpenCV worker threads to prevent CPU oversubscription.

Benchmark a short disposable training workload across batch sizes 16/32 and worker counts 0/2/4. Record samples/sec, peak allocated/reserved VRAM, RAM and available GPU telemetry. Select a stable configuration with headroom; effective batch size and optimizer settings must remain controlled between experiments. Use gradient accumulation only if necessary.

Use standard supported device settings. Do not overclock, disable protection, alter drivers or make system-wide changes. Report throttling or unavailable telemetry accurately.

Gate: an actual forward/backward/update test runs on the RTX 3050 without OOM, and performance measurements determine the loader/batch settings.

## Phase 6 — Establish controlled baselines

Run the following on locked development folds:

1. Constant prediction and the previous fixed azimuth rule, both evaluated on those same folds.
2. A small azimuth-only classifier using sin/cos; fit it exclusively on training partitions.
3. Raw grayscale image-only ResNet18: genuinely no azimuth input or canonicalization.
4. Canonicalized image-only ResNet18: illumination information enters preprocessing.
5. Raw-image ResNet18 plus azimuth MLP.
6. Canonicalized-image ResNet18 plus explicitly identified acquisition-angle MLP.

Image-only and azimuth-only systems are diagnostic ablations. Keep a model using both image and illumination for the final production pipeline, as required by the brief.

Start with a screening fold for implementation checks; confirm conclusions across all five folds. Use comparable training budgets, seeds and optimization settings. Export balanced accuracy, each class recall, confusion counts, ROC-AUC, log loss, threshold and train-evaluation gap. Report mean/std and angle-bin/pair-group performance, with group-based uncertainty estimates where feasible.

Gate: the multimodal model's benefit and remaining illumination-dependent errors are measured fairly. A high score triggers inspection, not automatic rejection or acceptance.

## Phase 7 — Improve transfer learning in a bounded search

Keep ResNet18 initially. Compare replicated three-channel grayscale with pretrained normalization against the current one-channel adaptation. For dataset normalization, estimate statistics from each training partition only. Neither approach is guaranteed to win.

Screen a small, recorded set of configurations:

- Head warm-up for roughly two epochs, then gradual backbone unfreezing.
- Backbone learning rate around 1e-5 to 3e-5, head/metadata learning rate around 1e-4 to 3e-4.
- Weight decay 1e-4 or 1e-3; classifier dropout 0.3 or 0.5.
- Mild physically consistent augmentation; retain 256x256 initially.
- Early-stopping patience around four epochs, total cap 25 epochs, chunks of at most five epochs.

Define BatchNorm behavior during freezing: frozen parameters alone do not freeze running statistics. Preserve optimizer/scheduler state when unfreezing and resuming. Use one imbalance strategy initially: fold-training-only BCE pos_weight. Do not stack weighted sampling and weighted loss without a controlled experiment.

Limit initial tuning to four configurations rather than a full Cartesian search. Confirm the strongest one or two across folds. Only test gated fusion/FiLM or a second backbone if diagnosed errors justify the extra experiment. If performance is weak, report it rather than repeatedly reusing the holdout until a score improves.

Gate: select by comparable held-out balanced accuracy, per-class recall, variability and shortcut diagnostics, with all experiment settings recorded.

## Phase 8 — Inspect errors and calibrate the selected pipeline

Create Grad-CAM and original/canonical image galleries for at least 20 fixed samples spanning both classes, correct/incorrect predictions, difficult angles and paired observations. Inspect rim/relief attention, shadow dependence and rotation borders. Treat saliency as diagnostic evidence, not proof of physical reasoning.

Select one final decision threshold from development calibration/out-of-fold predictions. Mark scores computed on those same predictions as threshold-selection estimates. If averaging models, calibrate the actual averaged-probability procedure; do not reuse a single-model threshold without checking calibration.

Freeze architecture, preprocessing, training schedule and threshold procedure, then evaluate the final holdout once. Export predictions and group-aware uncertainty. If it fails, document the failed result; further tuning makes that holdout development data and requires fresh evaluation evidence.

Gate: final quality claims come from a partition not used to choose that pipeline.

## Phase 9 — Retrain for submission and validate the CSV

After selecting the pipeline, fit a fresh final model on all eligible labeled observations, including earlier development/holdout rows, using a fixed schedule chosen from development experiments. Use a new run directory. Do not resume the old excluded-data model. State that the refitted model has no independent labeled evaluation after this refit; the earlier holdout score describes the selected procedure.

Reuse the preselected threshold without fitting it to the final model's in-sample training predictions. Record the calibration-transfer limitation. An ensemble is optional only if its grouped development evaluation and runtime justify it.

Run all 2,000 evaluation images through the chosen neural pipeline with their recorded sun angles. Use deterministic preprocessing and the checkpoint's inference precision. Never derive output labels from image IDs or exact-match label lookup.

Write the new submission under its run directory first. Validate:

- Columns exactly image_id,label, with no index or extra fields.
- Exactly one row per evaluation metadata record, in the same order.
- Exact ID preservation, unique IDs, no nulls and integer labels in {0,1}.
- Checkpoint reload reproduces the predictions.

Preserve the previous root submission before replacing it with the verified new result.

## Required handoff

Deliver runnable commands and a concise report with original/eligible/excluded counts; per-phase findings; runtime settings; fold scores; threshold-selection method; final holdout result; original-versus-new comparable metrics; remaining uncertainties; and output paths.

Provide CSVs for data audit, duplicate pairs, exclusions, splits, hardware benchmarks, training history, fold metrics, per-angle metrics, out-of-fold probabilities, holdout predictions, final probabilities, run summary and submission. Include source changes, passing test results, locked dependencies, checkpoints and the visual audit gallery.

Persist progress after every chunk. Continue through phases whose checks pass; if an evidence gap blocks one phase, finish independent preparation and describe the specific missing evidence. Do not invent labels, unavailable measurements or guaranteed accuracy gains.
