import json
import torch
import pathlib
import librosa
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .dataset import AudioDataset, extract_melspectrogram, extract_cqt
from .model import ShortChunkCNN
from ..commons import get_intervals
from ..mapping import artist_code_map

def parse_args():
    parser = argparse.ArgumentParser(description="DL-based singer classification training.")
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="directory containing training data")
    parser.add_argument("--output_dir", type=str, required=True, help="path to output directory")
    parser.add_argument("--sr", default=16000, type=int, help="sampling rate")
    parser.add_argument("--split_audio", action="store_true", help="whether to split audio into segments")
    parser.add_argument("--silent_threshold", default=30, type=int, help="silent threshold (in dB) for splitting audio")
    parser.add_argument("--min_seg", default=10.0, type=float, help="minimum segment length (in seconds) after splitting")
    parser.add_argument("--max_seg", default=15.0, type=float, help="maximum segment length (in seconds) after splitting")
    parser.add_argument("--max_silence", default=10.0, type=float, help="maximum silence length (in seconds) to keep in a segment after splitting")
    parser.add_argument("--time_stretch_ratio", default=0.7, type=float, help="probability of applying time stretching (only for training set)")
    parser.add_argument("--pitch_shift_ratio", default=0.7, type=float, help="probability of applying pitch shifting (only for training set)")
    parser.add_argument("--noise_injection_ratio", default=0.7, type=float, help="probability of applying noise injection (only for training set)")
    # Model parameters
    parser.add_argument("--used_spec", nargs="+", default=["mel", "cqt"], help="Type of spectrogram to use. Choose from 'mel' and 'cqt'.")
    parser.add_argument("--mel_n_channels", type=int, default=64, help="Number of channels for Mel spectrogram CNN.")
    parser.add_argument("--mel_n_fft", type=int, default=512, help="Number of FFT components for Mel spectrogram.")
    parser.add_argument("--mel_hop_length", type=int, default=160, help="Hop length for Mel spectrogram.")
    parser.add_argument("--mel_power", type=float, default=2.0, help="Exponent for the magnitude spectrogram for Mel spectrogram.")
    parser.add_argument("--mel_fmin", type=float, default=65.0, help="Minimum frequency for Mel spectrogram.")
    parser.add_argument("--mel_fmax", type=float, default=8000.0, help="Maximum frequency for Mel spectrogram.")
    parser.add_argument("--mel_n_mels", type=int, default=128, help="Number of Mel bands for Mel spectrogram.")
    parser.add_argument("--cqt_n_channels", type=int, default=64, help="Number of channels for CQT CNN.")
    parser.add_argument("--cqt_hop_length", type=int, default=512, help="Hop length for CQT.")
    parser.add_argument("--cqt_fmin", type=float, default=librosa.note_to_hz('C1'), help="Minimum frequency for CQT.")
    parser.add_argument("--cqt_n_bins", type=int, default=84, help="Number of frequency bins for CQT.")
    parser.add_argument("--cqt_bins_per_octave", type=int, default=12, help="Number of bins per octave for CQT.")
    # Training parameters
    parser.add_argument("--gpu", type=str, default="0", help="GPU id to use.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for the fully connected layer.")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="Minimum learning rate for scheduler.")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience based on validation accuracy.")
    return parser.parse_args()

def main():
    args = parse_args()

    data_dir = pathlib.Path(args.data_dir)
    train_names_json = data_dir / "train.json"
    val_names_json = data_dir / "val.json"
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(train_names_json, "r") as f:
        train_names = json.load(f)
    with open(val_names_json, "r") as f:
        val_names = json.load(f)
    
    train_files = [data_dir / name for name in train_names]
    val_files = [data_dir / name for name in val_names]

    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    )
    print(f"Using device: {device}")

    train_dataset = AudioDataset(
        data_type="train", used_spec=args.used_spec, audio_files=train_files, sr=args.sr, 
        split_audio=args.split_audio, silent_threshold=args.silent_threshold, min_seg=args.min_seg, 
        max_seg=args.max_seg, max_silence=args.max_silence, time_stretch_ratio=args.time_stretch_ratio, 
        pitch_shift_ratio=args.pitch_shift_ratio, noise_injection_ratio=args.noise_injection_ratio,
        mel_n_fft=args.mel_n_fft, mel_hop_length=args.mel_hop_length, mel_power=args.mel_power,
        mel_fmin=args.mel_fmin, mel_fmax=args.mel_fmax, mel_n_mels=args.mel_n_mels,
        cqt_hop_length=args.cqt_hop_length, cqt_fmin=args.cqt_fmin,
        cqt_n_bins=args.cqt_n_bins, cqt_bins_per_octave=args.cqt_bins_per_octave
    )
    val_dataset = AudioDataset(
        data_type="val", used_spec=args.used_spec, audio_files=val_files, sr=args.sr, 
        split_audio=args.split_audio, silent_threshold=args.silent_threshold, min_seg=args.min_seg, 
        max_seg=args.max_seg, max_silence=args.max_silence, mel_n_fft=args.mel_n_fft, 
        mel_hop_length=args.mel_hop_length, mel_power=args.mel_power,
        mel_fmin=args.mel_fmin, mel_fmax=args.mel_fmax, mel_n_mels=args.mel_n_mels,
        cqt_hop_length=args.cqt_hop_length, cqt_fmin=args.cqt_fmin,
        cqt_n_bins=args.cqt_n_bins, cqt_bins_per_octave=args.cqt_bins_per_octave
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    model = ShortChunkCNN(
        used_spec=args.used_spec, sr=args.sr, n_class=len(artist_code_map),
        mel_n_channels=args.mel_n_channels, cqt_n_channels=args.cqt_n_channels,
        dropout=args.dropout
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=80)
        for mels, cqts, labels in pbar:
            mels, cqts, labels = mels.to(device), cqts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(mels, cqts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            running_loss += loss.item() * labels.shape[0]

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch}/{args.epochs}, Training Loss: {epoch_loss:.4f}")

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for mels, cqts, labels in val_loader:
                mels, cqts, labels = mels.to(device), cqts.to(device), labels.to(device)
                outputs = model(mels, cqts)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.shape[0]

                probs = torch.nn.Softmax(dim=1)(outputs)
                _, predicted = torch.max(probs, 1)
                total += labels.shape[0]
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        print(f"Epoch {epoch}/{args.epochs}, Validation Loss: {val_loss:.4f}")
        print(f"Epoch {epoch}/{args.epochs}, Validation Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"New best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stopping_patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    print("===== Training Completed. =====\n")

    # Song-level evaluation
    top1 = 0
    top3 = 0
    val_preds = []
    val_labels = []
    for i in tqdm(range(len(val_files)), desc="Song-level Evaluation", ncols=80):
        file = val_files[i]
        label = str(file).split("/")[-3]
        label_id = artist_code_map[label]

        y, _ = librosa.load(path=file, sr=args.sr)
        y = librosa.util.normalize(y)

        intervals = get_intervals(
            y=y, sr=args.sr, split_audio=args.split_audio, silent_threshold=args.silent_threshold,
            min_seg=args.min_seg, max_seg=args.max_seg, max_silence=args.max_silence
        )

        probs_list = []
        for interval in intervals:
            y_segment = y[interval[0]:interval[1]]

            if len(y_segment) < int(args.sr * args.max_seg):
                y_segment = np.pad(y_segment, (0, int(args.sr * args.max_seg) - len(y_segment)), mode="constant")
            else:
                y_segment = y_segment[:int(args.sr * args.max_seg)]

            if "mel" in args.used_spec:
                mel = extract_melspectrogram(
                    y=y_segment, sr=args.sr, n_fft=args.mel_n_fft, hop_length=args.mel_hop_length,
                    power=args.mel_power, fmin=args.mel_fmin, fmax=args.mel_fmax, n_mels=args.mel_n_mels
                )
            else:
                mel = torch.zeros((1, 1, 1))  # dummy tensor
            
            if "cqt" in args.used_spec:
                cqt = extract_cqt(
                    y=y_segment, sr=args.sr, hop_length=args.cqt_hop_length, fmin=args.cqt_fmin,
                    n_bins=args.cqt_n_bins, bins_per_octave=args.cqt_bins_per_octave
                )
            else:
                cqt = torch.zeros((1, 1, 1))  # dummy tensor

            mel = mel.unsqueeze(0).to(device)
            cqt = cqt.unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(mel, cqt)
                probs = torch.nn.Softmax(dim=1)(outputs)
                probs_list.append(probs.cpu().numpy())
        
        probs_list = np.vstack(probs_list)
        avg_probs = np.mean(probs_list, axis=0)
        avg_probs = avg_probs / np.sum(avg_probs)
        top3_pred = np.argsort(avg_probs)[::-1][:3].tolist()
        top1_pred = top3_pred[0]
        val_preds.append(top1_pred)
        val_labels.append(label_id)

        if top1_pred == label_id:
            top1 += 1
        if label_id in top3_pred:
            top3 += 1

    print(f"Top-1 Accuracy: {top1 / len(val_files):.4f}")
    print(f"Top-3 Accuracy: {top3 / len(val_files):.4f}")

    # Save confusion matrix
    code_artist_map = {v: k for k, v in artist_code_map.items()}
    preds_artist = [code_artist_map[p] for p in val_preds]
    labels_artist = [code_artist_map[l] for l in val_labels]
    labels = list(artist_code_map.keys())
    cm_unnormalized = confusion_matrix(labels_artist, preds_artist, labels=labels, normalize=None)
    cm_normalized = confusion_matrix(labels_artist, preds_artist, labels=labels, normalize='true')
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