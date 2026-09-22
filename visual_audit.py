"""Qualitative model checks and predeclared illumination-shift diagnostic."""
from dataclasses import asdict
import json

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw

from audit_pipeline import partition
from dataset import LunarTerrainDataset
from experiment_utils import ROOT,atomic_csv,atomic_json,metrics
from training_engine import Settings,build_model,load_checkpoint,setup


def bootstrap_groups(out,pred,threshold):
    if (out/'holdout_uncertainty.csv').exists(): return
    groups=[]
    for _,g in pred.groupby('group'):
        p=g.probability.to_numpy()>=threshold; y=g.label.to_numpy()
        groups.append([((y==0)&(~p)).sum(),(y==0).sum(),((y==1)&p).sum(),(y==1).sum()])
    groups=np.asarray(groups); rng=np.random.default_rng(717); scores=[]
    for _ in range(1000):
        sums=groups[rng.integers(0,len(groups),len(groups))].sum(axis=0)
        if sums[1] and sums[3]: scores.append(.5*(sums[0]/sums[1]+sums[2]/sums[3]))
    atomic_csv(out/'holdout_uncertainty.csv',pd.DataFrame([dict(groups=len(groups),bootstrap_replicates=len(scores),
        lower_95=np.quantile(scores,.025),upper_95=np.quantile(scores,.975),unit='image_group')]))


def visual_audit(out,run_id,pred):
    if (out/'gradcam_gallery.jpg').exists(): return
    checkpoint=load_checkpoint(out/'models'/run_id/'inference.pt')
    settings=Settings(**checkpoint['settings']);setup(settings.seed)
    model=build_model(settings,False).cuda();model.load_state_dict(checkpoint['model']);model.eval()
    pred=pred.copy();pred['correct']=pred.label==pred.prediction
    # Include opposite-label same-image pairs, then round-robin the class/error
    # strata rather than letting sorted angle bins fill the entire gallery.
    pair_groups=[g for _,g in pred.groupby('pixel_hash') if g.label.nunique()>1]
    paired=pd.concat(pair_groups[:2]).head(4) if pair_groups else pred.head(0)
    representatives=pred.groupby(['label','correct','azimuth_bin']).head(1).copy()
    representatives['angle_priority']=representatives.azimuth_bin.map({0:0,4:1,2:2,6:3,1:4,5:5,3:6,7:7})
    candidates=representatives.sort_values(['angle_priority','image_id']).groupby(['label','correct'],sort=True)
    pools=[g.to_dict('records') for _,g in candidates]
    chosen=paired.to_dict('records'); seen=set(paired.image_id)
    for position in range(max(len(p) for p in pools)):
        for pool in pools:
            if position<len(pool) and pool[position]['image_id'] not in seen:
                chosen.append(pool[position]);seen.add(pool[position]['image_id'])
                if len(chosen)==20: break
        if len(chosen)==20: break
    selected=pd.DataFrame(chosen)
    if len(selected)<20:
        selected=pd.concat([selected,pred.sample(20,random_state=42)]).drop_duplicates('image_id').head(20)
    dataset=LunarTerrainDataset(selected,ROOT/'Train_DATA/train_images',False,asdict(settings))
    captured={}
    def hook(_module,_args,result):
        captured['features']=result;result.retain_grad()
    handle=model.image_branch.layer4.register_forward_hook(hook)
    canvas=Image.new('RGB',(3*256,len(selected)*285),'white');draw=ImageDraw.Draw(canvas)
    for i,(_,row) in enumerate(selected.iterrows()):
        batch=dataset[i];model.zero_grad(set_to_none=True)
        logits=model(batch['image'][None].cuda(),batch['azimuth'][None].cuda())
        (logits[0] if row.prediction else -logits[0]).backward()
        feature=captured['features']; grad=feature.grad
        cam=torch.relu((grad.mean((2,3),keepdim=True)*feature).sum(1))[0].detach().cpu().numpy()
        cam=cam/(cam.max()+1e-8);cam=cv2.resize(cam,(256,256))
        normalized=batch['image'].numpy()
        gray=np.clip((normalized*dataset.std+dataset.mean)[0]*255,0,255).astype(np.uint8)
        heat=cv2.cvtColor(cv2.applyColorMap((cam*255).astype(np.uint8),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
        overlay=(.5*gray[:,:,None]+.5*heat).astype(np.uint8)
        original=cv2.imread(str(ROOT/'Train_DATA/train_images'/row.image_id),cv2.IMREAD_GRAYSCALE)
        for col,picture in enumerate([np.repeat(original[:,:,None],3,axis=2),np.repeat(gray[:,:,None],3,axis=2),overlay]):
            canvas.paste(Image.fromarray(picture),(col*256,i*285))
        draw.text((4,i*285+258),f'{row.image_id} true={row.label} pred={row.prediction} p={row.probability:.3f} angle={row.sun_azimuth_angle}',fill='black')
    handle.remove();canvas.save(out/'gradcam_gallery.jpg')
    atomic_csv(out/'inspection_sample.csv',selected)


def angle_diagnostic(out,settings):
    if (out/'illumination_shift_metrics.csv').exists(): return
    from pipeline import run_to_completion,progress
    frame=pd.read_csv(out/'eligible.csv').query('partition=="development"')
    outer=frame[frame.azimuth_bin==6].copy()
    pool=frame[~frame.group.isin(outer.group)].copy()
    remainder,selection=partition(pool,8,891)
    fit,calibration=partition(remainder,7,892)
    frames=dict(fit=fit,selection=selection,calibration=calibration,outer=outer)
    for role,part in frames.items():
        if set(part.label)!={0,1}: raise ValueError('Angle diagnostic lacks both classes')
    assert not set(outer.group)&set(pool.group)
    manifest=pd.concat([part.assign(role=role) for role,part in frames.items()])
    atomic_csv(out/'illumination_shift_split.csv',manifest)
    progress(out,8,'Diagnostic: hold out 270-315 degrees and all related image groups from fitting')
    r=run_to_completion(out,'illumination_shift',settings,fold=-2,frames=frames)
    atomic_csv(out/'illumination_shift_metrics.csv',r)
