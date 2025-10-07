import json
import joblib
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ..commons import tqdm_joblib
from ..mapping import artist_code_map
from .utils import extract_features

def parse_args():
    parser = argparse.ArgumentParser(description="ML-based singer classification training.")
    # Data parameters
    parser.add_argument("--vocals_dir", required=True, type=str, help="input vocals files path")
    parser.add_argument("--inst_dir", type=str, default=None, help="input instrumental files path (if any)")
    parser.add_argument("--output_dir", required=True, type=str, help="output results path")
    parser.add_argument("--sr", default=16000, type=int, help="sampling rate")
    parser.add_argument("--split_audio", action="store_true", help="whether to split audio into segments")
    parser.add_argument("--used_features", default=["mfcc", "chroma", "beat"], nargs="+", type=str, help="list of features to use. Choose from 'mfcc', 'chroma', 'beat'")
    parser.add_argument("--silent_threshold", default=30, type=int, help="silent threshold (in dB) for splitting audio")
    parser.add_argument("--min_seg", default=10.0, type=float, help="minimum segment length (in seconds) after splitting")
    parser.add_argument("--max_seg", default=15.0, type=float, help="maximum segment length (in seconds) after splitting")
    parser.add_argument("--max_silence", default=10.0, type=float, help="maximum silence length (in seconds) to keep in a segment after splitting")
    parser.add_argument("--num_augments", default=0, type=int, help="number of augmentations per audio (only for training set)")
    parser.add_argument("--time_stretch_ratio", default=0.7, type=float, help="probability of applying time stretching (only for training set)")
    parser.add_argument("--pitch_shift_ratio", default=0.7, type=float, help="probability of applying pitch shifting (only for training set)")
    parser.add_argument("--noise_injection_ratio", default=0.7, type=float, help="probability of applying noise injection (only for training set)")
    # Model parameters
    parser.add_argument("--jobs", default=1, type=int, help="number of parallel jobs")
    parser.add_argument("--use_boosting", action="store_true", help="whether to use boosting (CatBoost) or single decision tree")
    parser.add_argument("--depth", default=2, type=int, help="depth of decision trees")
    parser.add_argument("--iters", default=10000, type=int, help="number of boosting iterations")
    parser.add_argument("--lr", default=0.05, type=float, help="learning rate")
    parser.add_argument("--l2_leaf_reg", default=5.0, type=float, help="L2 regularization term on weights")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    vocals_dir = pathlib.Path(args.vocals_dir)
    train_names_json = vocals_dir / "train.json"
    val_names_json = vocals_dir / "val.json"
    inst_dir = pathlib.Path(args.inst_dir) if args.inst_dir else None
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(train_names_json, "r") as f:
        train_names = json.load(f)
    with open(val_names_json, "r") as f:
        val_names = json.load(f)
    
    # Prepare dataset (features and labels)
    if args.jobs == 1:
        train_data = []
        for name in tqdm(train_names, desc="Extracting training features", ncols=80):
            features, labels = extract_features(
                data_type="train",
                vocals_path=vocals_dir / name,
                inst_path=inst_dir / name if inst_dir else None,
                sr=args.sr,
                split_audio=args.split_audio,
                silent_threshold=args.silent_threshold,
                min_seg=args.min_seg,
                max_seg=args.max_seg,
                max_silence=args.max_silence,
                num_augments=args.num_augments,
                time_stretch_ratio=args.time_stretch_ratio,
                pitch_shift_ratio=args.pitch_shift_ratio,
                noise_injection_ratio=args.noise_injection_ratio,
                used_features=args.used_features
            )
            train_data.append((features, labels))
    else:
        with tqdm_joblib(tqdm(desc="Extracting training features", total=len(train_names), ncols=80)):
            train_data = joblib.Parallel(n_jobs=args.jobs, verbose=0)(
                joblib.delayed(extract_features)(
                    data_type="train",
                    vocals_path=vocals_dir / name,
                    inst_path=inst_dir / name if inst_dir else None,
                    sr=args.sr,
                    split_audio=args.split_audio,
                    silent_threshold=args.silent_threshold,
                    min_seg=args.min_seg,
                    max_seg=args.max_seg,
                    max_silence=args.max_silence,
                    num_augments=args.num_augments,
                    time_stretch_ratio=args.time_stretch_ratio,
                    pitch_shift_ratio=args.pitch_shift_ratio,
                    noise_injection_ratio=args.noise_injection_ratio,
                    used_features=args.used_features
                ) for name in train_names
            )
    train_x, train_y = [], []
    for features, labels in train_data:
        train_x.extend(features)
        train_y.extend(labels)
    train_x, train_y = np.array(train_x), np.array(train_y)
    # Shuffle training data
    shuffled_train_indices = np.random.permutation(len(train_x))
    train_x = train_x[shuffled_train_indices]
    train_y = train_y[shuffled_train_indices]
    
    if args.jobs == 1:
        val_data = []
        for name in tqdm(val_names, desc="Extracting validation features", ncols=80):
            features, labels = extract_features(
                data_type="val",
                vocals_path=vocals_dir / name,
                inst_path=inst_dir / name if inst_dir else None,
                sr=args.sr,
                split_audio=args.split_audio,
                silent_threshold=args.silent_threshold,
                min_seg=args.min_seg,
                max_seg=args.max_seg,
                max_silence=args.max_silence,
                used_features=args.used_features
            )
            val_data.append((features, labels))
    else:
        with tqdm_joblib(tqdm(desc="Extracting validation features", total=len(val_names), ncols=80)):
            val_data = joblib.Parallel(n_jobs=args.jobs, verbose=0)(
                joblib.delayed(extract_features)(
                    data_type="val",
                    vocals_path=vocals_dir / name,
                    inst_path=inst_dir / name if inst_dir else None,
                    sr=args.sr,
                    split_audio=args.split_audio,
                    silent_threshold=args.silent_threshold,
                    min_seg=args.min_seg,
                    max_seg=args.max_seg,
                    max_silence=args.max_silence,
                    used_features=args.used_features
                ) for name in val_names
            )
    val_x, val_y, num_segs = [], [], []
    for features, labels in val_data:
        val_x.extend(features)
        val_y.extend(labels)
        num_segs.append(len(features))
    val_x, val_y = np.array(val_x), np.array(val_y)

    print(f"Training data shape: {train_x.shape}, Training labels shape: {train_y.shape}")
    print(f"Validation data shape: {val_x.shape}, Validation labels shape: {val_y.shape}")
    print(f"===== Start training classifier... =====")

    # Train classifier
    if not args.use_boosting:
        clf = DecisionTreeClassifier(
            max_depth=args.depth,
            random_state=args.seed
        )
    else:
        clf = CatBoostClassifier(
            depth=args.depth,
            learning_rate=args.lr,
            iterations=args.iters,
            l2_leaf_reg=args.l2_leaf_reg,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            random_seed=args.seed,
            verbose=200
        )
    clf.fit(train_x, train_y)
    print("===== Training completed. =====\n")

    # Save model
    if not args.use_boosting:
        output_model_path = output_dir / "model.joblib"
        joblib.dump(clf, output_model_path)
    else:
        output_model_path = output_dir / "model.cbm"
        clf.save_model(str(output_model_path))

    # Evaluate model
    val_pred_proba = clf.predict_proba(val_x)

    code_artist_map = {v: k for k, v in artist_code_map.items()}
    feature_idx = 0
    val_pred_proba_song = []
    val_y_song = []
    for n_segs in num_segs:
        proba = val_pred_proba[feature_idx:feature_idx+n_segs]
        avg_proba = np.mean(proba, axis=0)
        avg_proba = avg_proba / np.sum(avg_proba)  # Normalize
        val_pred_proba_song.append(avg_proba)
        val_y_song.append(val_y[feature_idx])
        feature_idx += n_segs
    
    val_pred_proba_song = np.array(val_pred_proba_song)
    val_y_song = np.array(val_y_song)

    top1_acc = top_k_accuracy_score(val_y_song, val_pred_proba_song, k=1, labels=list(artist_code_map.values()))
    top3_acc = top_k_accuracy_score(val_y_song, val_pred_proba_song, k=3, labels=list(artist_code_map.values()))
    print(f"Validation Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Validation Top-3 Accuracy: {top3_acc:.4f}\n")

    # Save confusion matrix
    val_y_pred_song = np.argmax(val_pred_proba_song, axis=1)

    code_artist_map = {v: k for k, v in artist_code_map.items()}
    val_y_pred_artist = [code_artist_map[code] for code in val_y_pred_song]
    val_y_artist = [code_artist_map[code] for code in val_y_song]
    labels = list(artist_code_map.keys())
    cm_unnormalized = confusion_matrix(val_y_artist, val_y_pred_artist, labels=labels, normalize=None)
    cm_normalized = confusion_matrix(val_y_artist, val_y_pred_artist, labels=labels, normalize='true')
    disp_unnormalized = ConfusionMatrixDisplay(confusion_matrix=cm_unnormalized, display_labels=labels)
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)

    _, ax = plt.subplots(figsize=(20, 20))
    disp_unnormalized.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True, values_format="d")
    plt.title("Confusion Matrix (Counts)\n", fontsize=26)
    ax.set_xlabel("Predicted label", fontsize=24)
    ax.set_ylabel("True label", fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=18)
    for text in disp_unnormalized.text_.ravel():
        text.set_fontsize(16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_cm_unnormalized_path = output_dir / "confusion_matrix_counts.png"
    plt.savefig(str(output_cm_unnormalized_path), dpi=300)
    plt.close()

    _, ax = plt.subplots(figsize=(20, 20))
    disp_normalized.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True, values_format=".2f")
    plt.title("Confusion Matrix (Normalized)\n", fontsize=26)
    ax.set_xlabel("Predicted label", fontsize=24)
    ax.set_ylabel("True label", fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=18)
    for text in disp_normalized.text_.ravel():
        text.set_fontsize(16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_cm_normalized_path = output_dir / "confusion_matrix_normalized.png"
    plt.savefig(str(output_cm_normalized_path), dpi=300)
    plt.close()

    print(f"===== Confusion matrices saved to folder: {output_dir} =====")

if __name__ == "__main__":
    main()