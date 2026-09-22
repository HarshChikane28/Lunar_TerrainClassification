"""Execute the audited experiment in restartable stages; all result tables are CSV."""
import argparse
import ctypes
import gc
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from experiment_utils import ROOT,DEFAULT_EXPERIMENT,atomic_csv,atomic_json,metrics,threshold_for,sha
from training_engine import (Settings,VARIANTS,setup,build_model,loader_for,shutdown,train_chunk,
    load_checkpoint,finalize_run,prediction_frame,save_checkpoint)


def progress(out,phase,message):
    atomic_json(out/'progress.json',dict(phase=phase,message=message,updated=time.strftime('%Y-%m-%d %H:%M:%S')))
    print(f'PHASE {phase}: {message}',flush=True)


def memory_mb():
    class MemoryStatus(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ['total','avail','page_total','page_avail','virt_total','virt_avail','extended']]
    obj=MemoryStatus(); obj.length=ctypes.sizeof(obj)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(obj))
    return (obj.total-obj.avail)/2**20


def benchmark(out):
    if (out/'runtime.json').exists(): return json.loads((out/'runtime.json').read_text())
    progress(out,5,'Benchmarking batch sizes and DataLoader worker counts on CUDA')
    setup(42)
    frame=pd.read_csv(out/'split_manifest.csv').query('fold==0 and role=="fit"').head(1024)
    results=[]
    for channels_last,batch in [(False,16),(False,32),(True,16),(True,32)]:
        for workers in [0,2,4]:
            settings=Settings(batch_size=batch,workers=workers,channels_last=channels_last)
            model=build_model(settings,False).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
            scaler=torch.amp.GradScaler('cuda'); loader=loader_for(frame,settings,True)
            torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter(); measured=None; count=0
            try:
                for step,data in enumerate(loader):
                    if step==3:
                        torch.cuda.synchronize(); measured=time.perf_counter()
                    x=data['image'].to(device='cuda',non_blocking=True,
                        memory_format=torch.channels_last if channels_last else torch.contiguous_format)
                    a=data['azimuth'].cuda(non_blocking=True); y=data['label'].cuda(non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.float16):
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x,a),y)
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                    if step>=3: count+=len(x)
                    if step==18: break
                torch.cuda.synchronize(); elapsed=time.perf_counter()-measured
                results.append(dict(batch_size=batch,workers=workers,channels_last=channels_last,samples_per_second=count/elapsed,
                    startup_seconds=measured-begin,allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                    reserved_mb=torch.cuda.max_memory_reserved()/2**20,system_used_ram_mb=memory_mb(),status='ok'))
            except torch.OutOfMemoryError:
                results.append(dict(batch_size=batch,workers=workers,channels_last=channels_last,status='oom'))
            finally:
                shutdown([loader]); del model,optimizer,scaler; gc.collect(); torch.cuda.empty_cache()
            print(results[-1],flush=True)
            atomic_csv(out/'hardware_benchmarks.csv',pd.DataFrame(results))
    data=pd.DataFrame(results); good=data[(data.status=='ok') & (data.reserved_mb<4800)]
    best=good.sort_values('samples_per_second',ascending=False).iloc[0]
    # Keep the selected batch size and memory layout identical across all experiments.
    runtime=dict(batch_size=int(best.batch_size),workers=int(best.workers),channels_last=bool(best.channels_last),gpu=torch.cuda.get_device_name(0),
                 torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,
                 telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used','--format=csv'],text=True))
    atomic_json(out/'runtime.json',runtime)
    versions=sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (out/'requirements-lock.txt').write_text('\n'.join(versions)+'\n',encoding='utf-8')
    return runtime


def baseline_features(df):
    a=np.deg2rad(df.sun_azimuth_angle.to_numpy())
    return np.stack([np.sin(a),np.cos(a)],axis=1)


def baselines(out):
    if (out/'baseline_metrics.csv').exists(): return
    progress(out,6,'Evaluating constant, fixed-angle rule and learned azimuth-only baselines on all five folds')
    manifest=pd.read_csv(out/'split_manifest.csv'); results=[]; predictions=[]
    for fold in range(5):
        frames={r:g for r,g in manifest[manifest.fold==fold].groupby('role')}
        fit,cal,outer=frames['fit'],frames['calibration'],frames['outer']
        clf=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=2,random_state=42)
        clf.fit(baseline_features(fit),fit.label,sample_weight=fit.sample_weight)
        threshold,_=threshold_for(cal.label,clf.predict_proba(baseline_features(cal))[:,1])
        for name,p,t in [('constant',np.zeros(len(outer)),.5),('fixed_angle_rule',(outer.angle.to_numpy()<270).astype(float),.5),
                         ('azimuth_only',clf.predict_proba(baseline_features(outer))[:,1],threshold)]:
            results.append(dict(fold=fold,variant=name,**metrics(outer.label,p,t)))
            pred=outer[['image_id','label','group','azimuth_bin']].copy()
            pred['fold']=fold; pred['variant']=name; pred['probability']=p; pred['threshold']=t; predictions.append(pred)
    atomic_csv(out/'baseline_metrics.csv',pd.DataFrame(results))
    atomic_csv(out/'baseline_oof.csv',pd.concat(predictions))


def run_to_completion(out,name,settings,fold,frames=None,finalize=True,threshold_override=None):
    run=out/'models'/name; checkpoint=run/'latest.pt'
    while True:
        if checkpoint.exists() and load_checkpoint(checkpoint)['complete']: break
        train_chunk(out,name,settings,fold,5,checkpoint if checkpoint.exists() else None,frames)
        gc.collect(); torch.cuda.empty_cache()
    if finalize: return finalize_run(out,name,fold,frames,threshold_override)
    return load_checkpoint(run/'best.pt')


def experiment_settings(runtime,variant,epochs=8,**extra):
    values=dict(VARIANTS[variant]); values.update(batch_size=runtime['batch_size'],workers=runtime['workers'],
        channels_last=runtime['channels_last'],total_epochs=epochs)
    values.update(extra)
    return Settings(**values)


def experiments(out,runtime):
    baselines(out)
    # Four primary ablations; four fusion configurations total (canonical, raw, opposite, gray).
    variants=['raw_image','canonical_image','raw_fusion','canonical_fusion','opposite_fusion','gray_fusion']
    records=[]
    for variant in variants:
        progress(out,6,f'Screening {variant} on fold 0; cap eight epochs with automatic early stopping')
        result=run_to_completion(out,f'{variant}_f0',experiment_settings(runtime,variant),0)
        records.append(dict(variant=variant,**result[result.role=='outer'].iloc[0].to_dict()))
        atomic_csv(out/'screening_metrics.csv',pd.DataFrame(records))
    # Select rotation/normalization on development evidence, never final holdout.
    fusion=pd.DataFrame(records).query('variant in ["raw_fusion","canonical_fusion","opposite_fusion","gray_fusion"]')
    selected=fusion.sort_values(['bacc','log_loss'],ascending=[False,True]).iloc[0].variant
    confirm=list(dict.fromkeys(['raw_image','canonical_image','raw_fusion','canonical_fusion',selected]))
    for variant in confirm:
        for fold in range(1,5):
            progress(out,6,f'Confirming {variant}, outer fold {fold+1}/5')
            run_to_completion(out,f'{variant}_f{fold}',experiment_settings(runtime,variant),fold)
    all_metrics=[]; oof=[]
    for variant in confirm:
        for fold in range(5):
            run=out/'models'/f'{variant}_f{fold}'
            m=pd.read_csv(run/'metrics.csv'); m['variant']=variant; all_metrics.append(m)
            p=pd.read_csv(run/'outer_predictions.csv'); p['variant']=variant; p['fold']=fold
            p['threshold']=float(m[m.role=='outer'].threshold.iloc[0]); oof.append(p)
    all_metrics=pd.concat(all_metrics,ignore_index=True); oof=pd.concat(oof,ignore_index=True)
    atomic_csv(out/'fold_metrics.csv',all_metrics); atomic_csv(out/'oof_predictions.csv',oof)
    summary=all_metrics[all_metrics.role=='outer'].groupby('variant').agg(mean_bacc=('bacc','mean'),std_bacc=('bacc','std'),mean_loss=('log_loss','mean')).reset_index()
    atomic_csv(out/'cv_summary.csv',summary)
    allowed=summary[summary.variant.str.contains('fusion')]
    selected=allowed.sort_values(['mean_bacc','mean_loss'],ascending=[False,True]).iloc[0].variant
    selected_oof=oof[oof.variant==selected]
    threshold,calibration_bacc=threshold_for(selected_oof.label,selected_oof.probability)
    epochs=int(np.median(all_metrics[(all_metrics.variant==selected)&(all_metrics.role=='outer')].best_epoch))
    atomic_json(out/'selection.json',dict(variant=selected,threshold=threshold,refit_epochs=max(3,epochs),
        oof_threshold_selection_bacc=calibration_bacc,notes='OOF threshold optimized here; final holdout evaluates frozen procedure. CV scores after variant selection are development estimates.'))
    return selected


def optimize(out,runtime):
    """One bounded regularization experiment, chosen on development folds only."""
    choice=json.loads((out/'selection.json').read_text()); variant=choice['variant']
    tuning=dict(classifier_dropout=.5,weight_decay=1e-3,backbone_lr=1e-5,head_lr=1e-4,total_epochs=16,patience=4)
    records=[]
    for fold in range(5):
        progress(out,7,f'Lower learning rate / stronger regularization confirmation fold {fold+1}/5')
        settings=experiment_settings(runtime,variant); settings=Settings(**(asdict(settings)|tuning))
        r=run_to_completion(out,f'tuned_{variant}_f{fold}',settings,fold)
        records.append(r[r.role=='outer'].iloc[0].to_dict())
        atomic_csv(out/'tuning_metrics.csv',pd.DataFrame(records))
    current=pd.read_csv(out/'cv_summary.csv').query('variant==@variant').iloc[0].mean_bacc
    tuned=float(pd.DataFrame(records).bacc.mean()); use_tuned=bool(tuned>current)
    prefix=f'tuned_{variant}' if use_tuned else variant
    oof=pd.concat([pd.read_csv(out/'models'/f'{prefix}_f{i}'/'outer_predictions.csv').assign(fold=i) for i in range(5)])
    t,b=threshold_for(oof.label,oof.probability)
    best_epochs=[load_checkpoint(out/'models'/f'{prefix}_f{i}'/'best.pt')['epoch'] for i in range(5)]
    settings=experiment_settings(runtime,variant)
    if use_tuned: settings=Settings(**(asdict(settings)|tuning))
    choice.update(tuned=use_tuned,settings=asdict(settings),threshold=t,refit_epochs=max(3,int(np.median(best_epochs))),
                  oof_threshold_selection_bacc=b,cv_mean=tuned if use_tuned else float(current),prefix=prefix)
    atomic_json(out/'selection.json',choice); atomic_csv(out/'selected_oof.csv',oof)
    return choice


def holdout_and_refit(out):
    from visual_audit import visual_audit, bootstrap_groups, angle_diagnostic
    choice=json.loads((out/'selection.json').read_text()); settings=Settings(**choice['settings'])
    progress(out,8,'Selected pipeline frozen; evaluate the untouched holdout once')
    # Holdout checkpoint selection and threshold calibration use only development subsets.
    result=run_to_completion(out,'holdout_assessment',settings,5,threshold_override=choice['threshold'])
    atomic_csv(out/'holdout_metrics.csv',result[result.role=='outer'])
    pred=pd.read_csv(out/'models/holdout_assessment/outer_predictions.csv')
    atomic_csv(out/'holdout_predictions.csv',pred)
    bootstrap_groups(out,pred,float(result[result.role=='outer'].threshold.iloc[0]))
    visual_audit(out,'holdout_assessment',pred)
    # This diagnostic is preregistered and does not influence final selection.
    angle_diagnostic(out,settings)
    progress(out,9,'Refitting the frozen model on all eligible labeled observations')
    eligible=pd.read_csv(out/'eligible.csv')
    settings.total_epochs=choice['refit_epochs']; settings.fixed_schedule=True
    frames={'fit':eligible}
    state=run_to_completion(out,'final_refit',settings,fold=-1,frames=frames,finalize=False)
    state['threshold']=choice['threshold']; state['calibration_role']='development_OOF_transfer_to_all_data_refit'
    save_checkpoint(out/'models/final_refit/inference.pt',state)
    from predict import predict
    output=out/'submission.csv'
    if not output.exists(): predict(out/'models/final_refit/inference.pt',output)
    if (ROOT/'submission.csv').exists() and not (out/'previous_submission.csv').exists():
        shutil.copy2(ROOT/'submission.csv',out/'previous_submission.csv')
    shutil.copy2(output,ROOT/'submission.csv')
    atomic_csv(out/'run_summary.csv',pd.DataFrame([dict(eligible_rows=len(eligible),final_epochs=state['epoch'],variant=choice['variant'],
        tuned=choice['tuned'],cv_bacc=choice['cv_mean'],holdout_bacc=float(result[result.role=='outer'].bacc.iloc[0]),
        threshold=choice['threshold'],refit_has_independent_test_labels=False,submission_rows=2000,
        gpu=torch.cuda.get_device_name(0),submission=str(output),checkpoint=str(out/'models/final_refit/inference.pt'))]))
    progress(out,'complete','Final submission validated; audit and quality evidence saved')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--experiment',type=Path,default=DEFAULT_EXPERIMENT)
    parser.add_argument('--stage',choices=['benchmark','baselines','experiments','optimize','finish','all'],default='all')
    args=parser.parse_args(); out=args.experiment
    if not (out/'split_manifest.csv').exists(): raise SystemExit('Run audit_pipeline.py first')
    from verify_inputs import verify
    verify(out)
    runtime=benchmark(out)
    if args.stage=='benchmark': return
    baselines(out)
    if args.stage=='baselines': return
    if args.stage in ['experiments','all']: experiments(out,runtime)
    if args.stage in ['optimize','all']: optimize(out,runtime)
    if args.stage in ['finish','all']: holdout_and_refit(out)
    from report_results import report
    report(out)


if __name__=='__main__': main()
