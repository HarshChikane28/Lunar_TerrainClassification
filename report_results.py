"""Regenerate a factual phase report from artifacts that actually exist."""
import argparse
import json
from pathlib import Path

import pandas as pd

from experiment_utils import DEFAULT_EXPERIMENT,atomic_csv


def markdown_table(df):
    values=df.copy()
    for c in values:
        if pd.api.types.is_float_dtype(values[c]): values[c]=values[c].map(lambda x:f'{x:.4f}')
    rows=[' | '.join(str(c) for c in values.columns),' | '.join('---' for _ in values.columns)]
    rows.extend(' | '.join(str(x) for x in row) for row in values.itertuples(index=False,name=None))
    return '\n'.join('| '+row+' |' for row in rows)


def report(out):
    out=Path(out)
    audit=json.loads((out/'audit_summary.json').read_text())
    eligible=pd.read_csv(out/'eligible.csv')
    split=json.loads((out/'split_fingerprint.json').read_text())
    progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
    completion_verified=(out/'completion_checks.csv').exists() and bool(pd.read_csv(out/'completion_checks.csv').passed.all())
    phases=[(0,'Evidence snapshot','original_snapshot/environment.json'),(1,'Joint-input audit','audit_summary.json'),
            (2,'Geometry implementation and visual comparison','geometry_gallery.jpg'),(3,'Grouped partitions','split_checks.csv'),
            (4,'Trainer correctness','test_results.txt'),(5,'GPU benchmark','runtime.json'),
            (6,'Five-fold ablations','cv_summary.csv'),(7,'Bounded tuning','tuning_metrics.csv'),
            (8,'Holdout and diagnostics','illumination_shift_metrics.csv'),(9,'Final refit/submission','run_summary.csv')]
    status=[]
    for phase,title,path in phases:
        present=(out/path).exists()
        state=('complete_verified' if completion_verified else 'artifact_available') if present else 'pending'
        if phase==2: state='implementation_verified; physical convention unresolved'
        if phase==7 and present:
            state='complete' if len(pd.read_csv(out/path))==5 else 'in_progress'
        status.append(dict(phase=phase,title=title,status=state,evidence=path))
    atomic_csv(out/'phase_status.csv',pd.DataFrame(status))
    pairs=pd.read_csv(out/'duplicate_pairs.csv')
    pair_summary=pd.DataFrame([dict(measure='same_pixels_opposing_labels',count=int((pairs.label_a!=pairs.label_b).sum())),
        dict(measure='angle_difference_at_most_1_degree',count=int((pairs.angle_delta<=1).sum())),
        dict(measure='angle_difference_at_most_5_degrees',count=int((pairs.angle_delta<=5).sum())),
        dict(measure='within_1_degree_of_opposite_lighting',count=int((pairs.angle_delta>=179).sum()))])
    atomic_csv(out/'duplicate_angle_summary.csv',pair_summary)
    text=['# Phased implementation and retraining report','',f'Current progress: {progress.get("message","audit prepared")}.','',
          f'Original training rows: {audit["original_rows"]}. Eligible: {audit["eligible_rows"]}. Quarantined: {audit["excluded_rows"]}.',
          f'Eligible rows contain {eligible.pixel_hash.nunique()} distinct decoded images; repeated images with different angles remain separate labeled observations.',
          f'Development: {split["development_rows"]}; untouched holdout: {split["holdout_rows"]}.',
          f'Exact image matches between train/evaluation: {audit["overlapping_test_images"]}; identical image-and-angle matches: {audit["overlapping_test_joint_inputs"]}.','',
          'The previous exclusion of every opposite-label image pair was too broad. Different-angle pairs are retained and grouped. The two identical-image/identical-angle contradictory observations remain quarantined; their labels were not changed.','',
          markdown_table(pd.DataFrame(status)), '', '## Duplicate-pair angle evidence', '',markdown_table(pair_summary),'',
          'Retaining different-angle pairs avoids the earlier blanket exclusion, but it does not certify that their angles or labels are physically correct. Most pairs are not approximately 180 degrees apart. No authoritative acquisition/label provenance was supplied.','']
    if (out/'baseline_metrics.csv').exists():
        b=pd.read_csv(out/'baseline_metrics.csv').groupby('variant').agg(mean_bacc=('bacc','mean'),std_bacc=('bacc','std')).reset_index()
        text+=['## Comparable metadata baselines','',markdown_table(b),'']
    for name,title in [('screening_metrics.csv','Single-fold screening (development evidence)'),('cv_summary.csv','Five-fold development comparison'),
                       ('tuning_metrics.csv','Tuning outer-fold results'),('holdout_metrics.csv','Untouched holdout assessment'),
                       ('holdout_uncertainty.csv','Holdout group bootstrap interval'),
                       ('holdout_comparison.csv','Same-holdout fixed-rule comparison (no reselection)'),
                       ('illumination_shift_metrics.csv','Preregistered 270-315 degree shift diagnostic'),
                       ('run_summary.csv','Final model and CSV'),('completion_checks.csv','Independent completion checks')]:
        if (out/name).exists():
            frame=pd.read_csv(out/name)
            keep=[c for c in ['variant','fold','role','best_epoch','eligible_rows','final_epochs','submission_rows',
                'tuned','bacc','mean_bacc','std_bacc','cv_bacc','holdout_bacc','recall_0','recall_1','auc',
                'threshold','lower_95','upper_95'] if c in frame]
            text+=['## '+title,'',markdown_table(frame[keep] if keep else frame),'']
    if (out/'fold_metrics.csv').exists():
        folds=pd.read_csv(out/'fold_metrics.csv')
        gaps=folds.pivot(index=['variant','fold'],columns='role',values='bacc').reset_index()
        gaps['fit_minus_outer']=gaps['fit']-gaps['outer']
        atomic_csv(out/'generalization_gaps.csv',gaps)
        text+=['## Matched-checkpoint, matched-threshold generalization gaps','',
               markdown_table(gaps.groupby('variant')[['fit','outer','fit_minus_outer']].mean().reset_index()),'']
    if (out/'runtime.json').exists():
        runtime=json.loads((out/'runtime.json').read_text())
        text+=['## Measured runtime configuration','',
               f'GPU: {runtime["gpu"]}; batch size: {runtime["batch_size"]}; workers: {runtime["workers"]}; channels-last: {runtime.get("channels_last",False)}; AMP: enabled. See hardware_benchmarks.csv for measured throughput and memory.','']
    if (out/'VISUAL_REVIEW.md').exists():
        text += [(out/'VISUAL_REVIEW.md').read_text(encoding='utf-8'),'']
    if (out/'VERIFICATION_NOTES.md').exists():
        text += [(out/'VERIFICATION_NOTES.md').read_text(encoding='utf-8'),'']
    if (out/'selection.json').exists() and (out/'holdout_metrics.csv').exists():
        choice=json.loads((out/'selection.json').read_text())
        baseline=pd.read_csv(out/'baseline_metrics.csv').query('variant=="fixed_angle_rule"').bacc.mean()
        holdout=pd.read_csv(out/'holdout_metrics.csv').iloc[0]
        text+=['## Quality conclusion','',
            f'The selected neural pipeline has {choice["cv_mean"]:.2%} mean development-fold balanced accuracy, compared with {baseline:.2%} for the fixed-angle rule on those folds. Its untouched holdout balanced accuracy is {holdout.bacc:.2%}. More elaborate training did not establish a gain over the simple angle baseline.','']
        if (out/'illumination_shift_metrics.csv').exists():
            shifted=pd.read_csv(out/'illumination_shift_metrics.csv').query('role=="outer"').iloc[0]
            text+=[f'The predeclared 270-315 degree held-out lighting diagnostic scored {shifted.bacc:.2%} balanced accuracy. This is a separate stress test, not the main holdout; it exposes a substantial limitation in lighting-distribution generalization. The final model was not retuned using either diagnostic result.','']
    text+=['## Limitations and interpretation','',
           '- The supplied brief does not specify a verified physical angle convention. Development results compare both signs and raw-image fusion; this is empirical model selection, not proof of the metadata geometry.',
           '- Near-duplicate search used a bounded perceptual-hash candidate search plus pixel MAE/correlation checks. It does not rule out every crop, transformation or shared acquisition scene. Scene IDs were not supplied.',
           '- Original imagery contains horizons/black background and large scene boundaries. Grad-CAM is diagnostic; it cannot prove that a classifier learned terrain geometry.',
           '- The old 0.7435 validation score and 0.7835 full-metadata rule score used different subsets and selection procedures. Neither is a controlled before/after comparison for this run.',
           '- Fold-wise thresholds are fitted on inner calibration groups. The final OOF threshold is a development selection estimate; the final holdout is evaluated only after the procedure is selected.',
           '- The final all-data refit has no independent labeled test set after refitting. Its threshold transfers from development OOF predictions and may shift in calibration.',
           '- Model predictions use image and angle; no evaluation labels or image-ID/exact-match label rules are used.','']
    path=out/'IMPLEMENTATION_REPORT.md';path.write_text('\n'.join(text),encoding='utf-8')
    print(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,default=DEFAULT_EXPERIMENT)
    report(parser.parse_args().experiment)
