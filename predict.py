"""Inference from a self-describing audited checkpoint; CSV schema enforced."""
import argparse
from pathlib import Path
import pandas as pd

from experiment_utils import ROOT, atomic_csv, validate_submission
from training_engine import Settings, setup, build_model, load_checkpoint, prediction_frame


def predict(checkpoint,output):
    state=load_checkpoint(checkpoint)
    if state.get('schema')!=2:
        raise ValueError('Legacy checkpoint: use its archived source/preprocessing in original_snapshot.')
    settings=Settings(**state['settings']); setup(settings.seed)
    model=build_model(settings,False).cuda(); model.load_state_dict(state['model'])
    frame=pd.read_csv(ROOT/'Test_DATA/test_metadata.csv')
    probabilities=prediction_frame(model,frame,settings,state['threshold'],ROOT/'Test_DATA/eval_images')
    submission=pd.DataFrame(dict(image_id=frame.image_id,label=probabilities.prediction.astype(int)))
    validate_submission(submission,frame.image_id)
    output=Path(output)
    if output.exists():
        raise FileExistsError(f'Preserve the existing output or choose a new path: {output}')
    atomic_csv(output,submission)
    atomic_csv(output.with_name('evaluation_probabilities.csv'),probabilities)
    print(f'Validated {len(submission)} predictions: {output}',flush=True)
    return submission


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); predict(args.checkpoint,args.output)


if __name__=='__main__': main()
