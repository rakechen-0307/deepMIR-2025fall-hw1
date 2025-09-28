import json
import torch
import pathlib
import librosa
import argparse
import numpy as np
from tqdm import tqdm

from .dataset import extract_melspectrogram, extract_cqt
from .model import ShortChunkCNN
from ..commons import get_intervals
from ..mapping import artist_code_map

def parse_args():
    parser = argparse.ArgumentParser(description="DL-based singer classification inference.")
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="input files path")
    parser.add_argument("--exp_dir", type=str, required=True, help="experiments/results path")
    parser.add_argument("--sr", default=16000, type=int, help="sampling rate")
    parser.add_argument("--split_audio", action="store_true", help="whether to split audio into segments")
    parser.add_argument("--silent_threshold", default=30, type=int, help="silent threshold (in dB) for splitting audio")
    parser.add_argument("--min_seg", default=10.0, type=float, help="minimum segment length (in seconds) after splitting")
    parser.add_argument("--max_seg", default=15.0, type=float, help="maximum segment length (in seconds) after splitting")
    parser.add_argument("--max_silence", default=10.0, type=float, help="maximum silence length (in seconds) to keep in a segment after splitting")
    # Model parameters
    parser.add_argument("--used_spec", nargs="+", default=["mel", "cqt"], help="Type of spectrogram to use. Choose from 'mel' and 'cqt'.")
    parser.add_argument("--mel_n_channels", type=int, default=128, help="Number of channels for Mel spectrogram CNN.")
    parser.add_argument("--mel_n_fft", type=int, default=512, help="Number of FFT components for Mel spectrogram.")
    parser.add_argument("--mel_hop_length", type=int, default=128, help="Hop length for Mel spectrogram.")
    parser.add_argument("--mel_power", type=float, default=2.0, help="Exponent for the magnitude spectrogram for Mel spectrogram.")
    parser.add_argument("--mel_fmin", type=float, default=65.0, help="Minimum frequency for Mel spectrogram.")
    parser.add_argument("--mel_fmax", type=float, default=8000.0, help="Maximum frequency for Mel spectrogram.")
    parser.add_argument("--mel_n_mels", type=int, default=128, help="Number of Mel bands for Mel spectrogram.")
    parser.add_argument("--cqt_n_channels", type=int, default=128, help="Number of channels for CQT CNN.")
    parser.add_argument("--cqt_hop_length", type=int, default=1024, help="Hop length for CQT.")
    parser.add_argument("--cqt_fmin", type=float, default=librosa.note_to_hz('C1'), help="Minimum frequency for CQT.")
    parser.add_argument("--cqt_n_bins", type=int, default=168, help="Number of frequency bins for CQT.")
    parser.add_argument("--cqt_bins_per_octave", type=int, default=24, help="Number of bins per octave for CQT.")
    # Other parameters
    parser.add_argument("--gpu", type=str, default="0", help="GPU id to use.")
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = pathlib.Path(args.data_dir)
    test_dir = data_dir / "test"
    exp_dir = pathlib.Path(args.exp_dir)
    ckpt_path = exp_dir / "best_model.pth"

    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    )
    print(f"Using device: {device}")

    model = ShortChunkCNN(
        used_spec=args.used_spec, sr=args.sr, n_class=len(artist_code_map),
        mel_n_channels=args.mel_n_channels, cqt_n_channels=args.cqt_n_channels
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()

    code_artist_map = {v: k for k, v in artist_code_map.items()}
    results = {}
    test_files = sorted([test_dir / f.name for f in test_dir.iterdir() if str(f).endswith('.mp3')])
    with torch.no_grad():
        for i in tqdm(range(len(test_files)), desc="Inference", ncols=80):
            file = test_files[i]
            y, _ = librosa.load(file, sr=args.sr)
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

                outputs = model(mel, cqt)
                probs = torch.nn.Softmax(dim=1)(outputs)
                probs_list.append(probs.cpu().numpy())
            
            probs_list = np.vstack(probs_list)
            avg_probs = np.mean(probs_list, axis=0)
            avg_probs = avg_probs / np.sum(avg_probs)
            top3_idx = np.argsort(avg_probs)[::-1][:3].tolist()
            top3_artist = [code_artist_map[idx] for idx in top3_idx]
            results[str(file.name).replace(".mp3", "")] = top3_artist
    
    # Save results
    with open(exp_dir / "test_pred.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()