"""Small persistence/metrics primitives shared by the audited experiment runner."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = ROOT / 'runs' / 'phasewise_v3'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, default=str).encode())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    def clean(item):
        if isinstance(item, dict): return {k:clean(v) for k,v in item.items()}
        if isinstance(item, (list,tuple)): return [clean(v) for v in item]
        if isinstance(item, (float,np.floating)) and not np.isfinite(item): return None
        return item
    temp.write_text(json.dumps(clean(value), indent=2, default=str, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def threshold_for(y, p):
    """O(n log n) exact sweep. Ties prefer the threshold closest to 0.5."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if set(y) != {0, 1} or not np.isfinite(p).all():
        raise ValueError('Threshold calibration needs both classes and finite probabilities.')
    order = np.argsort(p, kind='stable')
    y, p = y[order], p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(p)) + 1]
    candidates = np.r_[p[starts], np.nextafter(p[-1], np.inf), 0.5]
    left = np.searchsorted(p, candidates, side='left')
    pos = np.r_[0, np.cumsum(y)]
    neg = np.r_[0, np.cumsum(1-y)]
    scores = 0.5 * ((pos[-1]-pos[left])/pos[-1] + neg[left]/neg[-1])
    tied = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
    best = tied[np.argmin(np.abs(candidates[tied] - 0.5))]
    return float(candidates[best]), float(scores[best])


def metrics(y, p, threshold=0.5):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if len(y) == 0 or not np.isfinite(p).all():
        raise ValueError('Empty/nonfinite predictions.')
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return dict(n=len(y), bacc=float(balanced_accuracy_score(y, pred)) if both else None,
                recall_0=float(tn/(tn+fp)) if tn+fp else None,
                recall_1=float(tp/(tp+fn)) if tp+fn else None,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                auc=float(roc_auc_score(y, p)) if both else None,
                log_loss=float(log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])),
                threshold=float(threshold))


def validate_submission(frame, ids):
    if frame.columns.tolist() != ['image_id', 'label']:
        raise ValueError('Submission must have exactly image_id,label.')
    if frame.image_id.tolist() != list(ids) or frame.image_id.duplicated().any():
        raise ValueError('Submission image order/identity differs from evaluation metadata.')
    if frame.isna().any().any() or not pd.api.types.is_integer_dtype(frame.label):
        raise ValueError('Submission labels must be non-null integers.')
    if not set(frame.label).issubset({0, 1}):
        raise ValueError('Submission labels must be binary.')
