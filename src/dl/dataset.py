import torch
import librosa
import numpy as np
from random import random

from .utils import augment_audio
from ..commons import get_intervals
from ..mapping import artist_code_map

def extract_melspectrogram(
    y, sr=16000, n_fft=512, hop_length=160, power=2.0, fmin=65.0, 
    fmax=8000.0, n_mels=128
):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, power=power,
        fmin=fmin, fmax=fmax, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(S=mel, ref=np.max)

    # Convert to torch tensor with shape (1, n_mels, time)
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)

def extract_cqt(
    y, sr=16000, hop_length=512, fmin=librosa.note_to_hz('C1'), 
    n_bins=84, bins_per_octave=12
):
    cqt = librosa.cqt(
        y=y, sr=sr, hop_length=hop_length, fmin=fmin, 
        n_bins=n_bins, bins_per_octave=bins_per_octave
    )
    cqt_db = librosa.amplitude_to_db(S=np.abs(cqt), ref=np.max)

    # Convert to torch tensor with shape (1, n_bins, time)
    return torch.tensor(cqt_db, dtype=torch.float32).unsqueeze(0)

class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_type, used_spec, audio_files, sr=16000, split_audio=False,
        silent_threshold=30, min_seg=10.0, max_seg=15.0, max_silence=10.0,
        time_stretch_ratio=0.5, pitch_shift_ratio=0.5, noise_injection_ratio=0.5,
        mel_n_fft=512, mel_hop_length=160, mel_power=2.0, mel_fmin=65.0, mel_fmax=8000.0, 
        mel_n_mels=128, cqt_hop_length=512, cqt_fmin=librosa.note_to_hz('C1'), 
        cqt_n_bins=84, cqt_bins_per_octave=12
    ):
        assert data_type in ["train", "val"], "data_type must be 'train' or 'val'"

        self.sr = sr
        self.data_type = data_type
        self.used_spec = used_spec
        self.split_audio = split_audio
        self.audio_files = audio_files
        self.silent_threshold = silent_threshold
        self.min_seg = min_seg
        self.max_seg = max_seg
        self.max_silence = max_silence
        self.time_stretch_ratio = time_stretch_ratio
        self.pitch_shift_ratio = pitch_shift_ratio
        self.noise_injection_ratio = noise_injection_ratio

        # Mel Spectrogram parameters
        self.mel_n_fft = mel_n_fft
        self.mel_hop_length = mel_hop_length
        self.mel_power = mel_power
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.mel_n_mels = mel_n_mels

        # CQT parameters
        self.cqt_hop_length = cqt_hop_length
        self.cqt_fmin = cqt_fmin
        self.cqt_n_bins = cqt_n_bins
        self.cqt_bins_per_octave = cqt_bins_per_octave

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file = self.audio_files[idx]
        label = str(file).split("/")[-3]
        label_id = artist_code_map[label]

        y, _ = librosa.load(path=file, sr=self.sr)
        y = librosa.util.normalize(S=y)

        intervals = get_intervals(
            y=y, sr=self.sr, split_audio=self.split_audio, silent_threshold=self.silent_threshold,
            min_seg=self.min_seg, max_seg=self.max_seg, max_silence=self.max_silence
        )
        
        if self.data_type == "train":
            # random select one segment
            selected_interval = intervals[np.random.randint(0, len(intervals))]
            y = y[selected_interval[0]:selected_interval[1]]

            # augmentation
            time_stretch = True if random() < self.time_stretch_ratio else False
            pitch_shift = True if random() < self.pitch_shift_ratio else False
            noise_injection = True if random() < self.noise_injection_ratio else False
            y = augment_audio(
                y=y, sr=self.sr, time_stretch=time_stretch, pitch_shift=pitch_shift, 
                noise_injection=noise_injection
            )
        elif self.data_type == "val":
            # always select the first segment
            selected_interval = intervals[0]
            y = y[selected_interval[0]:selected_interval[1]]
        else:
            raise ValueError("data_type must be 'train' or 'val'")
        
        # pad or truncate to max_seg
        if len(y) < int(self.sr * self.max_seg):
            y = np.pad(y, (0, int(self.sr * self.max_seg) - len(y)), mode="constant")
        else:
            y = y[:int(self.sr * self.max_seg)]

        if "mel" in self.used_spec:
            mel = extract_melspectrogram(
                y=y, sr=self.sr, n_fft=self.mel_n_fft, hop_length=self.mel_hop_length,
                power=self.mel_power, fmin=self.mel_fmin, fmax=self.mel_fmax, n_mels=self.mel_n_mels,
            )
        else:
            mel = torch.zeros((1, 1, 1))  # dummy tensor
        
        if "cqt" in self.used_spec:
            cqt = extract_cqt(
                y=y, sr=self.sr, hop_length=self.cqt_hop_length, fmin=self.cqt_fmin,
                n_bins=self.cqt_n_bins, bins_per_octave=self.cqt_bins_per_octave,
            )
        else:
            cqt = torch.zeros((1, 1, 1))  # dummy tensor

        label_id = torch.tensor(label_id, dtype=torch.long)
        return mel, cqt, label_id