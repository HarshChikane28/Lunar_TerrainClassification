"""Illumination-aware input with deterministic per-sample/per-epoch augmentation."""
from pathlib import Path
import zlib

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def encode_azimuth(angle_degrees):
    radians=np.deg2rad(float(angle_degrees)%360)
    return torch.tensor([np.sin(radians),np.cos(radians)],dtype=torch.float32)


def rotate_to_canonical(image, azimuth_degrees, canonical_degrees=0.0, sign=-1, padding='constant'):
    """OpenCV positive displayed CCW rotation; dataset convention is experimental.

    sign=-1 assumes recorded angles increase counter-clockwise in the image plane.
    sign=+1 tests the opposite convention. Zero's geographic origin is unspecified.
    Metadata returned by the dataset is the ORIGINAL acquisition angle.
    """
    height,width=image.shape[:2]
    angle=(canonical_degrees+sign*(float(azimuth_degrees)%360))%360
    if angle==0: return image.copy()
    matrix=cv2.getRotationMatrix2D(((width-1)/2,(height-1)/2),angle,1.0)
    return cv2.warpAffine(image,matrix,(width,height),flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT if padding=='constant' else cv2.BORDER_REFLECT_101,borderValue=127)


class LunarTerrainDataset(Dataset):
    def __init__(self, metadata, image_dir, training=False, settings=None):
        self.metadata=pd.read_csv(metadata) if isinstance(metadata,(str,Path)) else metadata.reset_index(drop=True)
        required={'image_id','sun_azimuth_angle'}
        if not required.issubset(self.metadata): raise ValueError('Missing dataset columns')
        if training and 'label' not in self.metadata: raise ValueError('Training requires labels')
        if not np.isfinite(self.metadata.sun_azimuth_angle).all(): raise ValueError('Nonfinite angle')
        self.ids=self.metadata.image_id.astype(str).tolist()
        self.angles=self.metadata.sun_azimuth_angle.to_numpy(dtype=float)
        self.labels=self.metadata.label.to_numpy(dtype=np.float32) if 'label' in self.metadata else None
        self.weights=self.metadata.get('sample_weight',pd.Series(np.ones(len(self.metadata)))).to_numpy(dtype=np.float32)
        self.image_dir=Path(image_dir); self.training=training
        self.settings=settings or dict(channels=1,canonical=True,rotation_sign=-1,padding='reflect',seed=42)
        self.epoch=torch.zeros(1,dtype=torch.int64).share_memory_()
        self.mean=np.array([.485,.456,.406] if self.settings['channels']==3 else [.5],dtype=np.float32)[:,None,None]
        self.std=np.array([.229,.224,.225] if self.settings['channels']==3 else [.5],dtype=np.float32)[:,None,None]

    def __len__(self): return len(self.ids)

    def set_epoch(self,epoch): self.epoch[0]=epoch

    def __getitem__(self,index):
        cv2.setNumThreads(1)
        image=cv2.imread(str(self.image_dir/self.ids[index]),cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape!=(256,256): raise ValueError(f'Unreadable/wrong shape: {self.ids[index]}')
        if self.settings.get('canonical',True):
            image=rotate_to_canonical(image,self.angles[index],sign=self.settings.get('rotation_sign',-1),padding=self.settings.get('padding','constant'))
        image=image.astype(np.float32)/255.0
        if self.training:
            seed=(self.settings.get('seed',42)+int(self.epoch[0])*1000003+zlib.crc32(self.ids[index].encode()))%2**32
            rng=np.random.default_rng(seed)
            if rng.random()<.5:
                image=np.clip(image*rng.uniform(.9,1.1)+rng.uniform(-.08,.08),0,1)
        image=np.repeat(image[None],self.settings['channels'],axis=0)
        image=(image-self.mean)/self.std
        item=dict(image=torch.from_numpy(np.ascontiguousarray(image)),azimuth=encode_azimuth(self.angles[index]),image_id=self.ids[index])
        if self.labels is not None:
            item['label']=torch.tensor(self.labels[index]); item['weight']=torch.tensor(self.weights[index])
        return item
