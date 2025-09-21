import json
import joblib
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from catboost import CatBoostClassifier

from ..commons import tqdm_joblib
from .utils import extract_features

def parse_args():
    parser = argparse.ArgumentParser(description="ML-based singer classification training.")
    parser.add_argument("--input_dir", required=True, type=str, help="input audio files path")
    parser.add_argument("--output_dir", required=True, type=str, help="output results path")
    parser.add_argument("--sr", default=16000, type=int, help="sampling rate")
    parser.add_argument("--n_mfcc", default=13, type=int, help="number of MFCCs to extract")
    parser.add_argument("--len_segment", default=0, type=int, help="length of audio segments (in seconds)")
    parser.add_argument("--jobs", default=1, type=int, help="number of parallel jobs")
    parser.add_argument("--depth", default=3, type=int, help="depth of decision trees")
    parser.add_argument("--iters", default=2000, type=int, help="number of boosting iterations")
    parser.add_argument("--lr", default=0.05, type=float, help="learning rate")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = pathlib.Path(args.input_dir)
    train_names_json = input_dir / "train.json"
    val_names_json = input_dir / "val.json"
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(train_names_json, "r") as f:
        train_names = json.load(f)
    with open(val_names_json, "r") as f:
        val_names = json.load(f)
    
    # Prepare dataset (features and labels)
    with tqdm_joblib(tqdm(desc="Extracting training features", total=len(train_names), ncols=80)):
        train_data = joblib.Parallel(n_jobs=args.jobs, verbose=0)(
            joblib.delayed(extract_features)(
                file_path=input_dir / name,
                sr=args.sr,
                n_mfcc=args.n_mfcc,
                len_segment=args.len_segment
            ) for name in train_names
        )
    train_x, train_y = [], []
    for features, labels in train_data:
        train_x.extend(features)
        train_y.extend(labels)
    train_x, train_y = np.array(train_x), np.array(train_y)
    
    with tqdm_joblib(tqdm(desc="Extracting validation features", total=len(val_names), ncols=80)):
        val_data = joblib.Parallel(n_jobs=args.jobs, verbose=0)(
            joblib.delayed(extract_features)(
                file_path=input_dir / name,
                sr=args.sr,
                n_mfcc=args.n_mfcc,
                len_segment=args.len_segment
            ) for name in val_names
        )
    val_x, val_y = [], []
    for features, labels in val_data:
        val_x.extend(features)
        val_y.extend(labels)
    val_x, val_y = np.array(val_x), np.array(val_y)

    print(f"Training data shape: {train_x.shape}, Training labels shape: {train_y.shape}")
    print(f"Validation data shape: {val_x.shape}, Validation labels shape: {val_y.shape}")
    print(f"Start training classifier...")

    # Train classifier
    clf = CatBoostClassifier(
        iterations=args.iters,
        depth=args.depth,
        learning_rate=args.lr,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=args.seed,
        verbose=100
    )
    clf.fit(train_x, train_y, eval_set=(val_x, val_y), early_stopping_rounds=50)

    # Save model
    output_model_path = output_dir / "model.cbm"
    clf.save_model(str(output_model_path))

if __name__ == "__main__":
    main()