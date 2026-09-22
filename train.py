"""Explicit CLI for the audited GPU trainer. See pipeline.py for phase orchestration."""
import argparse
from pathlib import Path

from experiment_utils import DEFAULT_EXPERIMENT
from training_engine import Settings, VARIANTS, load_checkpoint, train_chunk, finalize_run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',type=Path,default=DEFAULT_EXPERIMENT)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--variant',choices=list(VARIANTS),default='canonical_fusion')
    parser.add_argument('--fold',type=int,choices=range(6),default=0)
    parser.add_argument('--total-epochs',type=int,default=12)
    parser.add_argument('--chunk-epochs',type=int,default=5)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    settings=Settings(**VARIANTS[args.variant],total_epochs=args.total_epochs)
    if args.resume:
        settings=Settings(**load_checkpoint(args.resume)['settings'])
        if settings.total_epochs!=args.total_epochs:
            parser.error('--total-epochs must match the resumed schedule')
    state=train_chunk(args.experiment,args.run_id,settings,args.fold,args.chunk_epochs,args.resume)
    if state['complete']:
        finalize_run(args.experiment,args.run_id,args.fold)


if __name__=='__main__': main()
