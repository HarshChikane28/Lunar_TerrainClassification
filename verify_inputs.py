"""Verify audited raw-file identities without rewriting data or split manifests."""
import argparse
from pathlib import Path

import pandas as pd

from experiment_utils import DEFAULT_EXPERIMENT,ROOT,atomic_csv,sha


def verify(experiment):
    rows=[]
    for name,directory in [('data_audit.csv','Train_DATA/train_images'),('test_audit.csv','Test_DATA/eval_images')]:
        frame=pd.read_csv(experiment/name)
        for record in frame.itertuples():
            path=ROOT/directory/record.image_id
            actual=sha(path.read_bytes()) if path.is_file() else 'missing'
            rows.append(dict(image_id=record.image_id,expected_hash=record.file_hash,actual_hash=actual,unchanged=actual==record.file_hash))
    result=pd.DataFrame(rows);atomic_csv(experiment/'input_verification.csv',result)
    if not result.unchanged.all(): raise ValueError('Audited image bytes changed; start a new audited experiment.')
    for source,name in [('Train_DATA/train_metadata.csv','data_audit.csv'),('Test_DATA/test_metadata.csv','test_audit.csv')]:
        current=pd.read_csv(ROOT/source);original=pd.read_csv(experiment/name)
        columns=current.columns.tolist()
        pd.testing.assert_frame_equal(current,original[columns],check_dtype=False,check_exact=True)
    print(f'Verified {len(result)} image identities and both metadata tables; source data unchanged.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,default=DEFAULT_EXPERIMENT)
    verify(parser.parse_args().experiment)
