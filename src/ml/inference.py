import json
import joblib
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from catboost import CatBoostClassifier

from ..commons import tqdm_joblib
from ..mapping import artist_code_map
from .utils import extract_features

def parse_args():
    parser = argparse.ArgumentParser(description="ML-based singer classification inference.")
    # Data parameters
    parser.add_argument("--vocals_dir", required=True, type=str, help="input vocals files path")
    parser.add_argument("--inst_dir", type=str, default=None, help="input instrumental files path (if any)")
    parser.add_argument("--exp_dir", required=True, type=str, help="experiments/results path")
    parser.add_argument("--sr", default=16000, type=int, help="sampling rate")
    parser.add_argument("--split_audio", action="store_true", help="whether to split audio into segments")
    parser.add_argument("--silent_threshold", default=30, type=int, help="silent threshold (in dB) for splitting audio")
    parser.add_argument("--min_seg", default=10.0, type=float, help="minimum segment length (in seconds) after splitting")
    parser.add_argument("--max_seg", default=15.0, type=float, help="maximum segment length (in seconds) after splitting")
    parser.add_argument("--max_silence", default=10.0, type=float, help="maximum silence length (in seconds) to keep in a segment after splitting")
    # Model parameters
    parser.add_argument("--jobs", default=1, type=int, help="number of parallel jobs")
    parser.add_argument("--use_boosting", action="store_true", help="whether to use boosting (CatBoost) or single decision tree")
    return parser.parse_args()

def main():
    args = parse_args()
    vocals_dir = pathlib.Path(args.vocals_dir)
    inst_dir = pathlib.Path(args.inst_dir) if args.inst_dir else None
    vocals_test_dir = vocals_dir / "test"
    inst_test_dir = inst_dir / "test" if inst_dir else None
    exp_dir = pathlib.Path(args.exp_dir)
    ckpt_path = exp_dir / "model.joblib" if not args.use_boosting else exp_dir / "model.cbm" 

    # Load checkpoint
    if not args.use_boosting:
        clf = joblib.load(ckpt_path)
    else:
        clf = CatBoostClassifier()
        clf.load_model(str(ckpt_path))

    test_files = sorted([f.name for f in vocals_test_dir.iterdir() if str(f).endswith('.mp3')])
    # Prepare dataset (features and labels)
    if args.jobs == 1:
        test_data = []
        for name in tqdm(test_files, desc="Extracting features", ncols=80):
            file_name, features = extract_features(
                data_type="test",
                vocals_path=vocals_test_dir / name,
                inst_path=inst_test_dir / name if inst_dir else None,
                sr=args.sr,
                split_audio=args.split_audio,
                silent_threshold=args.silent_threshold,
                min_seg=args.min_seg,
                max_seg=args.max_seg,
                max_silence=args.max_silence
            )
            test_data.append((file_name, features))
    else:
        with tqdm_joblib(tqdm(desc="Extracting features", total=len(test_files), ncols=80)):
            test_data = joblib.Parallel(n_jobs=args.jobs, verbose=0)(
                joblib.delayed(extract_features)(
                    data_type="test",
                    vocals_path=vocals_test_dir / name,
                    inst_path=inst_test_dir / name if inst_dir else None,
                    sr=args.sr,
                    split_audio=args.split_audio,
                    silent_threshold=args.silent_threshold,
                    min_seg=args.min_seg,
                    max_seg=args.max_seg,
                    max_silence=args.max_silence
                ) for name in test_files
            )
    names, x, num_segs = [], [], []
    for file_name, features in test_data:
        names.append(file_name)
        x.extend(features)
        num_segs.append(len(features))
    x = np.array(x)

    # Predict
    y_pred_proba = clf.predict_proba(x)

    # Save results
    code_artist_map = {v: k for k, v in artist_code_map.items()}
    results = {}
    feature_idx = 0
    for i, name in enumerate(names):
        proba = y_pred_proba[feature_idx:feature_idx+num_segs[i]]
        avg_proba = np.mean(proba, axis=0)
        avg_proba = avg_proba / np.sum(avg_proba)  # Normalize
        top3_idx = np.argsort(avg_proba)[::-1][:3].tolist()
        top3_artist = [code_artist_map[idx] for idx in top3_idx]
        results[name.replace(".mp3", "")] = top3_artist
        feature_idx += num_segs[i]

    # Save results
    with open(exp_dir / "test_pred.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()