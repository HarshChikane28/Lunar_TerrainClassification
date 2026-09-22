"""Independent post-run acceptance checks and full checkpoint prediction replay."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_utils import DEFAULT_EXPERIMENT, ROOT, atomic_csv, fingerprint, metrics, sha, validate_submission
from training_engine import Settings, build_model, load_checkpoint, metadata_signature, prediction_frame, setup, source_hash
from verify_inputs import verify


def verify_completion(out):
    out = Path(out)
    verify(out)
    checks = []
    def passed(name, detail):
        checks.append(dict(check=name, passed=True, detail=detail))
    audit = pd.read_csv(out/'data_audit.csv')
    eligible = pd.read_csv(out/'eligible.csv')
    excluded = pd.read_csv(out/'excluded_samples.csv')
    assert set(eligible.image_id).isdisjoint(excluded.image_id)
    assert set(eligible.image_id) | set(excluded.image_id) == set(audit.image_id)
    passed('source_row_accounting', f'{len(eligible)} eligible + {len(excluded)} quarantined = {len(audit)}')
    manifest = pd.read_csv(out/'split_manifest.csv')
    for fold, frame in manifest.groupby('fold'):
        assert frame.groupby('group').role.nunique().max() == 1
        assert not frame.image_id.duplicated().any()
        if fold < 5: assert set(frame.partition) == {'development'}
    outer = manifest.query('fold<5 and role=="outer"')
    assert not outer.image_id.duplicated().any()
    assert set(outer.image_id) == set(eligible.query('partition=="development"').image_id)
    passed('nested_group_splits', 'Five disjoint outer folds cover development exactly once; holdout excluded')
    oof = pd.read_csv(out/'selected_oof.csv')
    assert set(oof.image_id) == set(outer.image_id) and not oof.image_id.duplicated().any()
    passed('selected_oof_coverage', f'{len(oof)} unique development predictions')
    state = load_checkpoint(out/'models/final_refit/inference.pt')
    choice = json.loads((out/'selection.json').read_text())
    summary = pd.read_csv(out/'run_summary.csv').iloc[0]
    settings = Settings(**state['settings'])
    assert state['complete'] and settings.fixed_schedule and state['fold'] == -1
    assert state['epoch'] == choice['refit_epochs'] == summary.final_epochs
    assert summary.eligible_rows == len(eligible)
    assert state['threshold'] == choice['threshold']
    assert settings.use_metadata
    expected_signature=fingerprint(dict(settings=asdict(settings),fold=-1,
        data=metadata_signature(out),source=source_hash(),ids={'fit':eligible.image_id.tolist()}))
    assert state['signature']==expected_signature
    passed('final_refit_configuration', f'{state["epoch"]} fixed epochs, joint image/angle model, frozen OOF threshold')
    pair_metrics=[]
    for run in (out/'models').iterdir():
        if not (run/'latest.pt').exists(): continue
        latest = load_checkpoint(run/'latest.pt')
        history = pd.read_csv(run/'training_history.csv')
        assert latest['complete'], run.name
        assert history.epoch.tolist() == list(range(1, latest['epoch']+1)), run.name
        if (run/'outer_predictions.csv').exists():
            predictions=pd.read_csv(run/'outer_predictions.csv')
            threshold=float(pd.read_csv(run/'metrics.csv').query('role=="outer"').threshold.iloc[0])
            paired=predictions.groupby('pixel_hash').image_id.transform('size')>1
            for name,mask in [('repeated_image_observations',paired),('single_image_observations',~paired)]:
                subset=predictions[mask]
                if len(subset):
                    pair_metrics.append(dict(run_id=run.name,subset=name,
                        **metrics(subset.label,subset.probability,threshold)))
    atomic_csv(out/'pair_group_metrics.csv',pd.DataFrame(pair_metrics))
    passed('epoch_commit_histories', 'Every saved experiment completed; contiguous epochs without duplicates')
    meta = pd.read_csv(ROOT/'Test_DATA/test_metadata.csv')
    saved = pd.read_csv(out/'submission.csv')
    root = pd.read_csv(ROOT/'submission.csv')
    validate_submission(saved, meta.image_id)
    pd.testing.assert_frame_equal(saved, root)
    setup(settings.seed)
    model = build_model(settings, False).cuda()
    model.load_state_dict(state['model'])
    replay = prediction_frame(model, meta, settings, state['threshold'], ROOT/'Test_DATA/eval_images')
    probabilities = pd.read_csv(out/'evaluation_probabilities.csv')
    # Fast cuDNN/FP16 kernels can choose different algorithms across processes.
    # Require exact label identity; tolerate one FP16 epsilon for probabilities.
    tolerance=float(torch.finfo(torch.float16).eps) if settings.amp else 1e-5
    differences=pd.DataFrame(dict(image_id=meta.image_id,saved_probability=probabilities.probability,
        replay_probability=replay.probability,absolute_difference=np.abs(replay.probability-probabilities.probability),
        saved_label=saved.label,replay_label=replay.prediction))
    atomic_csv(out/'inference_replay_differences.csv',differences)
    max_difference=float(differences.absolute_difference.max())
    label_mismatches=int((saved.label!=replay.prediction).sum())
    atomic_csv(out/'inference_replay_summary.csv',pd.DataFrame([dict(rows=len(meta),
        label_mismatches=label_mismatches,max_absolute_probability_difference=max_difference,
        absolute_probability_tolerance=tolerance,precision='AMP_FP16' if settings.amp else 'FP32')]))
    np.testing.assert_array_equal(replay.prediction, saved.label)
    np.testing.assert_allclose(replay.probability, probabilities.probability, rtol=0, atol=tolerance)
    passed('full_gpu_inference_replay', f'All {len(meta)} labels reproduced; max probability drift {max_difference:.8f}, tolerance {tolerance:.8f}; not a bitwise probability claim')
    passed('submission_schema_and_root_copy', f'image_id,label; exact IDs/order; sha256={sha((out/"submission.csv").read_bytes())}')
    holdout = pd.read_csv(out/'holdout_predictions.csv')
    expected_holdout = set(eligible.query('partition=="holdout"').image_id)
    assert set(holdout.image_id) == expected_holdout
    rule = ((holdout.sun_azimuth_angle.to_numpy() % 360) < 270).astype(float)
    comparators = pd.DataFrame([
        dict(variant='selected_neural', **metrics(holdout.label, holdout.probability, choice['threshold'])),
        dict(variant='fixed_angle_rule', **metrics(holdout.label, rule)),
    ])
    atomic_csv(out/'holdout_comparison.csv', comparators)
    passed('sealed_holdout_coverage', f'{len(holdout)} rows; fixed-angle comparator is diagnostic only, no reselection')
    atomic_csv(out/'completion_checks.csv', pd.DataFrame(checks))
    print(pd.DataFrame(checks).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, default=DEFAULT_EXPERIMENT)
    verify_completion(parser.parse_args().experiment)
