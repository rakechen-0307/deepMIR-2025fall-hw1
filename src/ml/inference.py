import joblib
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from catboost import CatBoostClassifier

from ..commons import tqdm_joblib
from .utils import extract_features

def parse_args():
    parser = argparse.ArgumentParser(description="ML-based singer classification inference.")
    parser.add_argument("--input_dir", required=True, type=str, help="input audio files path")
    parser.add_argument("--exp_dir", required=True, type=str, help="experiments/results path")
    parser.add_argument("--sr", default=16000, type=int, help="sampling rate")
    parser.add_argument("--n_mfcc", default=13, type=int, help="number of MFCCs to extract")
    parser.add_argument("--len_segment", default=0, type=int, help="length of audio segments (in seconds)")
    parser.add_argument("--jobs", default=1, type=int, help="number of parallel jobs")
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = pathlib.Path(args.input_dir)
    input_test_dir = input_dir / "test"
    exp_dir = pathlib.Path(args.exp_dir)
    model_path = exp_dir / "model.cbm"
    result_dir = exp_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)    

    # Load model
    clf = CatBoostClassifier()
    clf.load_model(str(model_path))

    test_files = sorted([f.name for f in input_test_dir.iterdir() if str(f).endswith('.mp3')])
    # Prepare dataset (features and labels)
    with tqdm_joblib(tqdm(desc="Extracting features", total=len(test_files), ncols=80)):
        data = joblib.Parallel(n_jobs=args.jobs, verbose=0)(
            joblib.delayed(extract_features)(
                file_path=input_test_dir / name,
                sr=args.sr,
                n_mfcc=args.n_mfcc,
                len_segment=args.len_segment,
                return_label=False
            ) for name in test_files
        )
    x = []
    for features in data:
        x.extend(features)
    x = np.array(x)

    # Predict
    y_pred_proba = clf.predict_proba(x)
    
    # save top3 predictions
    top3 = np.argsort(y_pred_proba, axis=1)[:,-3:][:,::-1]  # (num_samples, 3)
    with open(result_dir / "top3_predictions.txt", "w") as f:
        for i, name in enumerate(test_files):
            preds = top3[i]
            preds_str = " ".join([str(p) for p in preds])
            f.write(f"{name} {preds_str}\n")

if __name__ == "__main__":
    main()