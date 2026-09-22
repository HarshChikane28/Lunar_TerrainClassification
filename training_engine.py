"""GPU training with committed epoch checkpoints and independent threshold calibration."""
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from dataset import LunarTerrainDataset
from experiment_utils import ROOT, atomic_csv, atomic_json, fingerprint, metrics, sha, threshold_for
from model import LunarFusionModel


@dataclass
class Settings:
    channels: int = 3
    canonical: bool = True
    rotation_sign: int = -1
    padding: str = 'constant'
    use_metadata: bool = True
    metadata_dropout: float = .1
    classifier_dropout: float = .3
    seed: int = 42
    batch_size: int = 16
    workers: int = 2
    amp: bool = True
    total_epochs: int = 12
    warmup_epochs: int = 2
    backbone_lr: float = 3e-5
    head_lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 4
    min_delta: float = .001
    pretrained: bool = True
    fixed_schedule: bool = False
    deterministic: bool = False
    channels_last: bool = True


VARIANTS = {
    'raw_image': dict(canonical=False,use_metadata=False),
    'canonical_image': dict(canonical=True,use_metadata=False),
    'raw_fusion': dict(canonical=False,use_metadata=True),
    'canonical_fusion': dict(canonical=True,use_metadata=True),
    'opposite_fusion': dict(canonical=True,use_metadata=True,rotation_sign=1),
    'gray_fusion': dict(canonical=True,use_metadata=True,channels=1),
}


def setup(seed,deterministic=False):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required; CUDA unavailable.')
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=not deterministic
    torch.backends.cudnn.deterministic=deterministic
    torch.use_deterministic_algorithms(deterministic)
    return torch.device('cuda')


def build_model(settings, pretrained=None):
    model=LunarFusionModel(pretrained=settings.pretrained if pretrained is None else pretrained,
        metadata_dropout=settings.metadata_dropout,classifier_dropout=settings.classifier_dropout,
        channels=settings.channels,use_metadata=settings.use_metadata)
    return model.to(memory_format=torch.channels_last) if settings.channels_last else model


def loader_for(frame,settings,training=False,image_dir=None):
    ds=LunarTerrainDataset(frame,image_dir or ROOT/'Train_DATA/train_images',training,asdict(settings))
    generator=torch.Generator().manual_seed(settings.seed)
    kwargs=dict(batch_size=settings.batch_size,shuffle=False,num_workers=settings.workers,
                pin_memory=True,generator=generator)
    if training:
        # Worker base-seed draws must never advance the sample-order generator.
        kwargs['sampler']=RandomSampler(ds,generator=torch.Generator().manual_seed(settings.seed))
    if settings.workers:
        kwargs.update(persistent_workers=True,prefetch_factor=2)
    return DataLoader(ds,**kwargs)


def shutdown(loaders):
    for loader in loaders:
        iterator=getattr(loader,'_iterator',None)
        if iterator is not None: iterator._shutdown_workers()
    gc.collect(); torch.cuda.empty_cache()


def forward_epoch(model,loader,settings,criterion=None,optimizer=None,scaler=None):
    training=optimizer is not None
    model.train(training)
    if training and not any(p.requires_grad for p in model.image_branch.parameters()):
        model.image_branch.eval()  # frozen BatchNorm statistics as well as parameters
    probabilities=[]; losses=0.0; count=0
    for batch in loader:
        x=batch['image'].to(device='cuda',non_blocking=True,
            memory_format=torch.channels_last if settings.channels_last else torch.contiguous_format)
        a=batch['azimuth'].cuda(non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast('cuda',dtype=torch.float16,enabled=settings.amp):
                logits=model(x,a)
                if criterion is not None:
                    labels=batch['label'].cuda(non_blocking=True)
                    weights=batch['weight'].cuda(non_blocking=True)
                    loss=(criterion(logits,labels)*weights).mean()
            if training:
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer); scaler.update()
        if criterion is not None: losses+=float(loss.detach())*len(x)
        count+=len(x)
        probabilities.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
    return losses/max(count,1),np.concatenate(probabilities)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def save_checkpoint(path,state):
    path=Path(path); temp=path.with_suffix('.tmp')
    torch.save(state,temp); os.replace(temp,path)


def load_checkpoint(path):
    # Only local checkpoints produced by this repository: includes Python RNG tuples.
    return torch.load(path,map_location='cpu',weights_only=False)


def source_hash():
    return fingerprint({p:sha((ROOT/p).read_bytes()) for p in ['dataset.py','model.py','training_engine.py','experiment_utils.py']})


def metadata_signature(experiment):
    return fingerprint({p:sha((ROOT/p).read_bytes()) for p in ['Train_DATA/train_metadata.csv','Test_DATA/test_metadata.csv']} |
        {'manifest':sha((Path(experiment)/'split_manifest.csv').read_bytes()),
         'audit':sha((Path(experiment)/'data_audit.csv').read_bytes())})


def train_chunk(experiment,run_id,settings,fold=0,chunk_epochs=5,resume=None,frames_override=None):
    if chunk_epochs<1 or settings.total_epochs<1 or settings.batch_size<1 or settings.workers<0:
        raise ValueError('Invalid training limits')
    experiment=Path(experiment); run=experiment/'models'/run_id
    setup(settings.seed,settings.deterministic)
    if frames_override is None:
        manifest=pd.read_csv(experiment/'split_manifest.csv')
        manifest=manifest[manifest.fold==fold]
        frames={role:g.reset_index(drop=True) for role,g in manifest.groupby('role')}
    else: frames=frames_override
    if 'fit' not in frames or set(frames['fit'].label)!={0,1}: raise ValueError('Training requires both classes')
    signature=fingerprint(dict(settings=asdict(settings),fold=fold,data=metadata_signature(experiment),source=source_hash(),
        ids={r:g.image_id.tolist() for r,g in frames.items()}))
    state=load_checkpoint(resume) if resume else None
    if state and state['signature']!=signature: raise ValueError('Resume incompatible with configuration, source, or data/splits')
    if not state and (run/'latest.pt').exists(): raise ValueError('Existing run: supply explicit resume path or a new run ID')
    run.mkdir(parents=True,exist_ok=True)
    atomic_json(run/'config.json',dict(settings=asdict(settings),fold=fold,signature=signature))
    model=build_model(settings,pretrained=False if state else None).cuda()
    head=list(model.metadata_branch.parameters())+list(model.classifier.parameters())
    optimizer=torch.optim.AdamW([dict(params=model.image_branch.parameters(),lr=settings.backbone_lr),
                                 dict(params=head,lr=settings.head_lr)],weight_decay=settings.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=settings.total_epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=settings.amp)
    history=[]; best_score=-1.; best_epoch=0; bad_epochs=0; start=1
    if state:
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng']); history=state['history']; start=state['epoch']+1
        best_score=state['best_score']; best_epoch=state['best_epoch']; bad_epochs=state['bad_epochs']
        atomic_csv(run/'training_history.csv',pd.DataFrame(history))
        # Repair an interruption after committing latest but before committing best.
        if state['epoch']==state['best_epoch']:
            save_checkpoint(run/'best.pt',state)
        if state['complete']: return state
    counts=frames['fit'].groupby('label').sample_weight.sum()
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([counts[0]/counts[1]],device='cuda'),reduction='none')
    eval_settings=Settings(**asdict(settings)); eval_settings.workers=0
    loaders={'fit':loader_for(frames['fit'],settings,True)}
    if 'selection' in frames:
        loaders['selection']=loader_for(frames['selection'],eval_settings,False)
    fit_eval=loader_for(frames['fit'],eval_settings,False)
    # Loader seeds are epoch-derived and do not depend on whether earlier epochs ran in this process.
    stop=min(settings.total_epochs,start+chunk_epochs-1)
    try:
        for epoch in range(start,stop+1):
            started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            for p in model.image_branch.parameters(): p.requires_grad_(epoch>settings.warmup_epochs)
            loaders['fit'].dataset.set_epoch(epoch)
            loaders['fit'].sampler.generator.manual_seed(settings.seed+epoch*1009)
            loaders['fit'].generator.manual_seed(settings.seed+epoch*1013)
            online_loss,online_p=forward_epoch(model,loaders['fit'],settings,criterion,optimizer,scaler)
            train_loss,train_p=forward_epoch(model,fit_eval,settings,criterion)
            train_score=metrics(frames['fit'].label,train_p)['bacc']
            if not settings.fixed_schedule:
                selection_loss,selection_p=forward_epoch(model,loaders['selection'],settings,criterion)
                score=metrics(frames['selection'].label,selection_p)['bacc']
            else: selection_loss=float('nan'); score=float('nan')
            scheduler.step()
            improved=settings.fixed_schedule or score>best_score+settings.min_delta
            if improved: best_score=score; best_epoch=epoch; bad_epochs=0
            else: bad_epochs+=1
            complete=epoch>=settings.total_epochs or (not settings.fixed_schedule and bad_epochs>=settings.patience)
            row=dict(epoch=epoch,online_train_loss=online_loss,train_eval_loss=train_loss,train_eval_bacc=train_score,
                     selection_loss=selection_loss,selection_bacc=score,gap=train_score-score,threshold=.5,
                     backbone_lr=optimizer.param_groups[0]['lr'],head_lr=optimizer.param_groups[1]['lr'],
                     seconds=time.perf_counter()-started,gpu_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                     gpu_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
            history.append(row)
            state=dict(schema=2,signature=signature,settings=asdict(settings),fold=fold,model=model.state_dict(),
                       optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                       rng=rng_state(),epoch=epoch,best_score=best_score,best_epoch=best_epoch,bad_epochs=bad_epochs,
                       threshold=.5,history=history,complete=complete)
            # latest.pt is the committed epoch; CSV is recoverable from it.
            save_checkpoint(run/'latest.pt',state)
            if improved: save_checkpoint(run/'best.pt',state)
            atomic_csv(run/'training_history.csv',pd.DataFrame(history))
            atomic_json(run/'status.json',dict(epoch=epoch,best_epoch=best_epoch,best_score=best_score,complete=complete))
            print(f'{run_id} epoch {epoch}: train_eval={train_score:.4f} selection={score:.4f} best={best_score:.4f} seconds={row["seconds"]:.1f}',flush=True)
            if complete: break
    finally:
        shutdown(list(loaders.values())+[fit_eval])
    return state


def prediction_frame(model,frame,settings,threshold=.5,image_dir=None):
    cfg=Settings(**asdict(settings)); cfg.workers=0
    loader=loader_for(frame,cfg,False,image_dir)
    _,p=forward_epoch(model,loader,settings)
    keep=[c for c in ['image_id','label','sun_azimuth_angle','group','pixel_hash','azimuth_bin'] if c in frame]
    out=frame[keep].copy(); out['probability']=p; out['prediction']=(p>=threshold).astype(int)
    shutdown([loader]); return out


def finalize_run(experiment,run_id,fold,frames_override=None,threshold_override=None):
    run=Path(experiment)/'models'/run_id
    if (run/'metrics.csv').exists(): return pd.read_csv(run/'metrics.csv')
    state=load_checkpoint(run/'best.pt'); settings=Settings(**state['settings']); setup(settings.seed)
    model=build_model(settings,False).cuda(); model.load_state_dict(state['model']); model.eval()
    manifest=pd.read_csv(Path(experiment)/'split_manifest.csv')
    frames=frames_override or {r:g.reset_index(drop=True) for r,g in manifest[manifest.fold==fold].groupby('role')}
    calibration=prediction_frame(model,frames['calibration'],settings)
    threshold,_=threshold_for(calibration.label,calibration.probability)
    if threshold_override is not None: threshold=float(threshold_override)
    state['threshold']=threshold
    state['calibration_role']='development_OOF_fixed_before_holdout' if threshold_override is not None else 'inner_calibration_not_outer_validation'
    save_checkpoint(run/'inference.pt',state)
    results=[]
    for role in ['fit','selection','calibration','outer']:
        pred=calibration if role=='calibration' else prediction_frame(model,frames[role],settings,threshold)
        pred['prediction']=(pred.probability>=threshold).astype(int)
        atomic_csv(run/f'{role}_predictions.csv',pred)
        results.append(dict(run_id=run_id,fold=fold,role=role,best_epoch=state['epoch'],**metrics(pred.label,pred.probability,threshold)))
    output=pd.DataFrame(results); atomic_csv(run/'metrics.csv',output)
    atomic_csv(run/'angle_metrics.csv',pd.DataFrame([dict(azimuth_bin=k,**metrics(g.label,g.probability,threshold))
        for k,g in pred.groupby('azimuth_bin')]))
    # Reloaded inference artifact must produce the same deterministic probabilities.
    reloaded=load_checkpoint(run/'inference.pt'); check=build_model(settings,False).cuda(); check.load_state_dict(reloaded['model'])
    pcheck=prediction_frame(check,frames['outer'].head(32),settings,threshold)
    np.testing.assert_allclose(pcheck.probability,pred.probability.iloc[:32],rtol=1e-5,atol=1e-5)
    print(f'{run_id}: outer bacc={output[output.role=="outer"].bacc.iloc[0]:.4f} threshold={threshold:.4f}',flush=True)
    return output
