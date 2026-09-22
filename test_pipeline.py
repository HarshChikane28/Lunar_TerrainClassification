"""Contract tests, including GPU epoch-boundary resume equivalence."""
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from dataset import LunarTerrainDataset,encode_azimuth,rotate_to_canonical
from experiment_utils import DEFAULT_EXPERIMENT,ROOT,threshold_for,validate_submission
from training_engine import Settings,build_model,train_chunk,load_checkpoint


class Contracts(unittest.TestCase):
    def test_threshold_exact(self):
        rng=np.random.default_rng(9)
        for _ in range(30):
            y=np.r_[0,1,rng.integers(0,2,40)]
            p=np.round(rng.uniform(size=len(y)),2)
            threshold,score=threshold_for(y,p)
            candidates=np.r_[np.unique(p),np.nextafter(p.max(),np.inf),.5]
            expected=max(balanced_accuracy_score(y,p>=t) for t in candidates)
            self.assertAlmostEqual(score,expected)
            self.assertAlmostEqual(score,balanced_accuracy_score(y,p>=threshold))

    def test_geometry_and_wrapping(self):
        im=np.zeros((9,9),np.uint8); im[4,7]=255
        rotated=rotate_to_canonical(im,90,sign=-1)
        self.assertEqual(np.unravel_index(rotated.argmax(),rotated.shape),(7,4))
        np.testing.assert_array_equal(rotate_to_canonical(im,360),im)
        np.testing.assert_array_equal(rotate_to_canonical(im,-90),rotate_to_canonical(im,270))
        np.testing.assert_allclose(encode_azimuth(360),encode_azimuth(0),atol=1e-7)

    def test_joint_policy_and_splits(self):
        audit=pd.read_csv(DEFAULT_EXPERIMENT/'data_audit.csv')
        joint=audit.groupby(['pixel_hash','angle']).label.transform('nunique')>1
        np.testing.assert_array_equal(joint,audit.excluded)
        self.assertTrue((audit.pixel_hash.duplicated(False)&~audit.excluded).any())
        splits=pd.read_csv(DEFAULT_EXPERIMENT/'split_manifest.csv')
        for fold,g in splits.groupby('fold'):
            self.assertEqual(g.groupby('group').role.nunique().max(),1)
            for _,r in g.groupby('role'): self.assertEqual(set(r.label),{0,1})
            if fold<5: self.assertEqual(set(g.partition),{'development'})

    def test_augmentation_replay(self):
        frame=pd.read_csv(DEFAULT_EXPERIMENT/'eligible.csv').head(8)
        dataset=LunarTerrainDataset(frame,ROOT/'Train_DATA/train_images',True,asdict(Settings(channels=1)))
        dataset.set_epoch(3); a=dataset[0]['image'].clone()
        _=dataset[1]; dataset.set_epoch(7); dataset.set_epoch(3)
        torch.testing.assert_close(a,dataset[0]['image'],rtol=0,atol=0)
        self.assertEqual(a.shape,(1,256,256))

    def test_shapes(self):
        for channels in [1,3]:
            model=build_model(Settings(channels=channels,pretrained=False))
            model.eval()
            with torch.no_grad(): output=model(torch.zeros(2,channels,256,256),torch.zeros(2,2))
            self.assertEqual(output.shape,(2,))

    def test_submission(self):
        valid=pd.DataFrame({'image_id':['a','b'],'label':[0,1]})
        validate_submission(valid,['a','b'])
        with self.assertRaises(ValueError): validate_submission(valid,['b','a'])
        with self.assertRaises(ValueError): validate_submission(valid.assign(label=[0,2]),['a','b'])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA is required')
    def test_gpu_resume_equivalence(self):
        self.check_resume(0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA is required')
    def test_gpu_persistent_workers_resume_equivalence(self):
        self.check_resume(4)

    def check_resume(self, workers):
        frame=pd.read_csv(DEFAULT_EXPERIMENT/'eligible.csv').groupby('label').head(16).reset_index(drop=True)
        frames={'fit':frame,'selection':frame.iloc[:16].copy()}
        frames['selection']=frame.groupby('label').head(4).reset_index(drop=True)
        settings=Settings(channels=1,workers=workers,total_epochs=2,warmup_epochs=0,pretrained=False,
                          deterministic=True,batch_size=8,patience=10)
        import shutil
        with tempfile.TemporaryDirectory(dir=DEFAULT_EXPERIMENT,prefix='resume-test-') as folder:
            root=Path(folder)
            for name in ['split_manifest.csv','data_audit.csv']:
                shutil.copy2(DEFAULT_EXPERIMENT/name,root/name)
            one=train_chunk(root,'continuous',settings,chunk_epochs=2,frames_override=frames)
            train_chunk(root,'resumed',settings,chunk_epochs=1,frames_override=frames)
            two=train_chunk(root,'resumed',settings,chunk_epochs=1,resume=root/'models/resumed/latest.pt',frames_override=frames)
            for key,value in one['model'].items():
                torch.testing.assert_close(value,two['model'][key],rtol=0,atol=0)
            self.assertEqual(one['scheduler'],two['scheduler'])
            self.assertEqual(one['best_epoch'],two['best_epoch'])
            incompatible=Settings(**asdict(settings)); incompatible.head_lr*=2
            with self.assertRaises(ValueError):
                train_chunk(root,'resumed',incompatible,resume=root/'models/resumed/latest.pt',frames_override=frames)


if __name__=='__main__': unittest.main(verbosity=2)
