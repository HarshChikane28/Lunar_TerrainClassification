"""Phases 0-3: immutable evidence, joint-input audit and nested grouped splits."""
import argparse
import importlib.metadata
import itertools
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedGroupKFold

from experiment_utils import ROOT, DEFAULT_EXPERIMENT, atomic_csv, atomic_json, fingerprint, sha


def snapshot(out):
    folder = out / 'original_snapshot'
    if folder.exists():
        return
    folder.mkdir(parents=True)
    for path in list(ROOT.glob('*.py')) + list(ROOT.glob('*.md')) + [ROOT/'requirements.txt', ROOT/'.gitignore', ROOT/'submission.csv']:
        if path.is_file():
            shutil.copy2(path, folder/path.name)
    for name in ['Train_DATA/train_metadata.csv', 'Test_DATA/test_metadata.csv']:
        shutil.copy2(ROOT/name, folder/Path(name).name)
    for command, filename in [(['git','diff','--binary'], 'initial.diff'), (['git','status','--short'], 'git_status.txt')]:
        (folder/filename).write_bytes(subprocess.check_output(command, cwd=ROOT))
    atomic_json(folder/'environment.json', dict(python=sys.version, executable=sys.executable,
        revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))


def inspect(out):
    rows, gray_images = [], {}
    for source, directory, csv in [('train','Train_DATA/train_images','Train_DATA/train_metadata.csv'),
                                    ('test','Test_DATA/eval_images','Test_DATA/test_metadata.csv')]:
        meta = pd.read_csv(ROOT/csv)
        required = {'image_id','sun_azimuth_angle'} | ({'label'} if source == 'train' else set())
        if not required.issubset(meta) or meta.image_id.duplicated().any() or meta[list(required)].isna().any().any():
            raise ValueError(f'Invalid metadata {csv}')
        if not np.isfinite(meta.sun_azimuth_angle).all():
            raise ValueError('Nonfinite azimuth')
        if source == 'train' and not set(meta.label).issubset({0,1}):
            raise ValueError('Invalid label')
        for item in meta.to_dict('records'):
            path = ROOT/directory/item['image_id']
            raw = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None or image.shape[:2] != (256,256) or image.dtype != np.uint8:
                raise ValueError(f'Invalid image: {path}')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            h = sha(gray.tobytes())
            gray_images.setdefault(h, gray)
            item.update(source=source, angle=float(item['sun_azimuth_angle'])%360,
                        file_hash=sha(raw), pixel_hash=h, shape=str(image.shape),
                        channels_equal=bool(image.ndim == 2 or np.all(image == image[:,:,:1])))
            rows.append(item)
    frame = pd.DataFrame(rows)
    train = frame[frame.source == 'train'].copy()
    joint_counts = train.groupby(['pixel_hash','angle']).label.transform('nunique')
    train['excluded'] = joint_counts > 1
    train['reason'] = np.where(train.excluded, 'identical_pixels_and_recorded_angle_opposing_labels', '')
    pairs=[]
    for h,g in train.groupby('pixel_hash', sort=False):
        for a,b in itertools.combinations(g.to_dict('records'),2):
            delta=abs((a['angle']-b['angle']+180)%360-180)
            pairs.append(dict(image_a=a['image_id'], image_b=b['image_id'], pixel_hash=h,
                              angle_a=a['angle'],angle_b=b['angle'],angle_delta=delta,
                              label_a=a['label'],label_b=b['label'],
                              category=('same_joint_input' if delta == 0 else 'different_illumination')))
    # Candidate discovery via 4 independent 16-bit pHash bands; all <=3-bit pairs share a band.
    unique=train.pixel_hash.unique().tolist()
    buckets=defaultdict(list); hashes={}; candidates=set()
    for h in unique:
        dct=cv2.dct(cv2.resize(gray_images[h],(32,32)).astype(np.float32))[:8,:8].ravel()
        bits=dct>np.median(dct[1:]); bits[0]=False
        value=int.from_bytes(np.packbits(bits).tobytes(),'big'); hashes[h]=value
        for band in range(4):
            key=(band,(value>>(band*16))&65535)
            for other in buckets[key]:
                if (value^hashes[other]).bit_count()<=3:
                    candidates.add(tuple(sorted((h,other))))
            buckets[key].append(h)
    parent={h:h for h in unique}
    def find(h):
        while parent[h]!=h:
            parent[h]=parent[parent[h]]; h=parent[h]
        return h
    near=[]
    for a,b in sorted(candidates):
        ia,ib=gray_images[a].astype(float),gray_images[b].astype(float)
        mae=float(np.abs(ia-ib).mean())
        corr=float(np.corrcoef(ia.ravel(),ib.ravel())[0,1]) if ia.std() and ib.std() else 0.0
        grouped=bool(mae<=5 and corr>=0.995)
        if grouped: parent[find(b)]=find(a)
        near.append(dict(hash_a=a,hash_b=b,hamming=(hashes[a]^hashes[b]).bit_count(),mae=mae,
                         correlation=corr,grouped=grouped))
    train['group']=train.pixel_hash.map(find)
    train['label']=train.label.astype(int)
    train['azimuth_bin']=(train.angle//45).astype(int)
    train['joint_repeats']=train.groupby(['pixel_hash','angle']).image_id.transform('size')
    # Exactly repeated same-label observations get total loss mass one within their group.
    train['sample_weight']=1/train.joint_repeats
    atomic_csv(out/'data_audit.csv', train)
    atomic_csv(out/'excluded_samples.csv', train[train.excluded])
    atomic_csv(out/'duplicate_pairs.csv', pd.DataFrame(pairs))
    atomic_csv(out/'near_duplicate_candidates.csv', pd.DataFrame(near, columns=['hash_a','hash_b','hamming','mae','correlation','grouped']))
    test=frame[frame.source=='test'].drop(columns='label')
    atomic_csv(out/'test_audit.csv',test)
    overlaps=test[['image_id','pixel_hash','angle']].merge(train[['image_id','pixel_hash','angle','label']],on='pixel_hash',suffixes=('_test','_train'))
    overlaps['same_joint_input']=overlaps.angle_test==overlaps.angle_train
    atomic_csv(out/'overlap_audit.csv',overlaps)
    atomic_csv(out/'class_azimuth_counts.csv',train.groupby(['excluded','label','azimuth_bin']).size().reset_index(name='count'))
    summary=dict(original_rows=len(train),eligible_rows=int((~train.excluded).sum()),excluded_rows=int(train.excluded.sum()),
                 image_pair_count=len(pairs),near_candidates=len(near),near_grouped=sum(n['grouped'] for n in near),
                 evaluation_rows=len(test),overlapping_test_images=int(overlaps.image_id_test.nunique()),
                 overlapping_test_joint_inputs=int(overlaps[overlaps.same_joint_input].image_id_test.nunique()),
                 rgb_channels_equal=int(frame.channels_equal.sum()),total_images=len(frame),
                 content_fingerprint=fingerprint(frame[['image_id','file_hash','angle','label']].fillna('').to_dict('records')))
    atomic_json(out/'audit_summary.json',summary)
    print(json.dumps(summary),flush=True)
    gallery_geometry(out,train,gray_images)
    return train[~train.excluded].copy().reset_index(drop=True)


def gallery_geometry(out,train,images):
    selected=train.groupby(['label','azimuth_bin'],sort=True).head(1)
    paired=train[train.pixel_hash.duplicated(False)].head(4)
    selected=pd.concat([selected,paired]).drop_duplicates('image_id').head(20)
    canvas=Image.new('RGB',(4*200,len(selected)*220),'white'); draw=ImageDraw.Draw(canvas)
    for row,(_,item) in enumerate(selected.iterrows()):
        im=images[item.pixel_hash]
        variants=[im]+[cv2.warpAffine(im,cv2.getRotationMatrix2D((127.5,127.5),s*item.angle,1),(256,256),
                     borderMode=cv2.BORDER_CONSTANT,borderValue=127) for s in [-1,1]]
        variants.append(cv2.warpAffine(im,cv2.getRotationMatrix2D((127.5,127.5),-item.angle,1),(256,256),borderMode=cv2.BORDER_REFLECT_101))
        for col,picture in enumerate(variants):
            canvas.paste(Image.fromarray(picture).convert('RGB').resize((195,195)),(col*200,row*220))
            text=f'{item.image_id} y={item.label} a={item.angle}' if col==0 else ['','-angle gray','+angle gray','-angle reflect'][col]
            draw.text((col*200,row*220+198),text,fill='black')
    canvas.save(out/'geometry_gallery.jpg')
    atomic_csv(out/'geometry_gallery_samples.csv',selected[['image_id','label','angle','pixel_hash']])


def strata(df):
    result=df.label.astype(str)+'_'+df.azimuth_bin.astype(str)
    return result if result.value_counts().min()>=7 else df.label


def partition(df,n,seed):
    a,b=next(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(df,strata(df),df.group))
    return df.iloc[a].copy(), df.iloc[b].copy()


def make_splits(out,eligible):
    dev,holdout=partition(eligible,7,42)
    records=[]
    for fold,(a,b) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=43).split(dev,strata(dev),dev.group)):
        outer_train,outer=dev.iloc[a],dev.iloc[b]
        remainder,selection=partition(outer_train,8,100+fold)
        fit,calibration=partition(remainder,7,200+fold)
        for role,part in [('fit',fit),('selection',selection),('calibration',calibration),('outer',outer)]:
            records.extend(dict(image_id=i,fold=fold,role=role) for i in part.image_id)
    # Separate development selection/calibration for the one-time holdout evaluation.
    remainder,selection=partition(dev,8,301)
    fit,calibration=partition(remainder,7,302)
    for role,part in [('fit',fit),('selection',selection),('calibration',calibration),('outer',holdout)]:
        records.extend(dict(image_id=i,fold=5,role=role) for i in part.image_id)
    eligible['partition']=np.where(eligible.image_id.isin(holdout.image_id),'holdout','development')
    atomic_csv(out/'eligible.csv',eligible)
    manifest=pd.DataFrame(records).merge(eligible,on='image_id',validate='many_to_one')
    checks=[]
    for fold,g in manifest.groupby('fold'):
        assert g.image_id.nunique()==len(eligible) if fold==5 else g.image_id.nunique()==len(dev)
        assert g.groupby('group').role.nunique().max()==1
        for role,part in g.groupby('role'):
            assert set(part.label)=={0,1}
            checks.append(dict(fold=fold,role=role,rows=len(part),groups=part.group.nunique(),both_classes=True,group_leakage=False))
    atomic_csv(out/'split_manifest.csv',manifest)
    atomic_csv(out/'split_checks.csv',pd.DataFrame(checks))
    atomic_csv(out/'split_summary.csv',manifest.groupby(['fold','role','label','azimuth_bin']).size().reset_index(name='count'))
    atomic_json(out/'split_fingerprint.json',dict(sha256=sha((out/'split_manifest.csv').read_bytes()),seed=42,
        holdout_rows=len(holdout),development_rows=len(dev),notes='Five outer folds; separate inner selection/calibration; fold 5 reserved for holdout.'))
    print(f'Splits: {len(dev)} development / {len(holdout)} untouched holdout',flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--experiment',type=Path,default=DEFAULT_EXPERIMENT)
    args=parser.parse_args(); out=args.experiment
    out.mkdir(parents=True,exist_ok=True); snapshot(out)
    if (out/'split_manifest.csv').exists():
        raise SystemExit('Audit/splits already exist; use a new experiment directory to change inputs.')
    cv2.setNumThreads(1)
    make_splits(out,inspect(out))


if __name__=='__main__': main()
