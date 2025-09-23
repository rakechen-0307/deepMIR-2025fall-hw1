import librosa
import numpy as np
from random import random

from ..mapping import artist_code_map

def augment_audio(y_vocals, y_inst, sr, time_stretch=False, pitch_shift=False, noise_injection=False):
    """
    Apply data augmentation techniques to the audio signal.
    """
    if time_stretch:
        rate = np.random.uniform(0.9, 1.1)
        y_vocals = librosa.effects.time_stretch(y=y_vocals, rate=rate)
        if y_inst is not None:
            y_inst = librosa.effects.time_stretch(y=y_inst, rate=rate)

    if pitch_shift:
        n_steps = np.random.randint(-2, 3)
        y_vocals = librosa.effects.pitch_shift(y=y_vocals, sr=sr, n_steps=n_steps)
        if y_inst is not None:
            y_inst = librosa.effects.pitch_shift(y=y_inst, sr=sr, n_steps=n_steps)

    if noise_injection:
        noise = np.random.normal(0, 0.002, len(y_vocals))
        y_vocals = y_vocals + noise
        if y_inst is not None:
            noise = np.random.normal(0, 0.002, len(y_inst))
            y_inst = y_inst + noise

    if y_inst is not None:
        return y_vocals, y_inst
    else:
        return y_vocals

def extract_features(
    vocals_path, inst_path=None, sr=16000, split_audio=False,
    silent_threshold=50, min_seg=10.0, max_silence=10.0, data_type="train", num_augments=0,
    time_stretch_ratio=0.5, pitch_shift_ratio=0.5, noise_injection_ratio=0.5
):
    """
    Extract audio features from a given file. 
    """
    y_vocals, _ = librosa.load(vocals_path, sr=sr)
    y_vocals = librosa.util.normalize(y_vocals)  # Normalize audio
    if inst_path:
        y_inst, _ = librosa.load(inst_path, sr=sr)
        y_inst = librosa.util.normalize(y_inst)  # Normalize audio

    if data_type == "train" or data_type == "val":
        label = str(vocals_path).split("/")[-3]
        label_id = artist_code_map[label]

    if data_type == "train" or data_type == "val":
        features, labels = [], []
    else:   
        features = []

    if data_type == "train":
        num_samples = num_augments + 1  # original + augmentations
    else:
        num_samples = 1  # only original

    for i in range(num_samples):
        if split_audio:
            non_silences = librosa.effects.split(y_vocals, top_db=silent_threshold)
            silences = []
            for j in range(len(non_silences) - 1):
                silence_start = non_silences[j][1]
                silence_end = non_silences[j + 1][0]
                silences.append(silence_end - silence_start)
            intervals = []
            curr_start = 0
            for j, silence in enumerate(silences):
                if silence / sr >= max_silence:
                    curr_end = non_silences[j][1]
                    if (curr_end - curr_start) / sr >= min_seg:
                        intervals.append([curr_start, curr_end])
                    curr_start = non_silences[j + 1][0]
            # Check for the last segment
            if (non_silences[-1][1] - curr_start) / sr >= min_seg:
                intervals.append([curr_start, non_silences[-1][1]])
            intervals = np.array(intervals)
        else:
            intervals = np.array([[0, len(y_vocals)]])

        for interval in intervals:
            start, end = interval
            vocals_segment = y_vocals[start:end]
            if inst_path:
                inst_segment = y_inst[start:end]

            if i != 0:
                time_stretch = True if random() < time_stretch_ratio else False
                pitch_shift = True if random() < pitch_shift_ratio else False
                noise_injection = True if random() < noise_injection_ratio else False
                if inst_path:
                    vocals_segment, inst_segment = augment_audio(vocals_segment, inst_segment, sr, time_stretch, pitch_shift, noise_injection)
                else:
                    vocals_segment = augment_audio(vocals_segment, None, sr, time_stretch, pitch_shift, noise_injection)

            # ========== Vocals MFCCs ==========
            vocals_mfccs = librosa.feature.mfcc(y=vocals_segment, sr=sr, n_mfcc=13)
            vocals_mfcc_mean = np.mean(vocals_mfccs, axis=1)
            vocals_mfcc_std = np.std(vocals_mfccs, axis=1)
            vocals_mfcc_vector = np.concatenate([vocals_mfcc_mean, vocals_mfcc_std])

            # ========== Instrumental MFCCs ==========
            if inst_path:
                inst_mfccs = librosa.feature.mfcc(y=inst_segment, sr=sr, n_mfcc=5)
                inst_mfcc_mean = np.mean(inst_mfccs, axis=1)
                inst_mfcc_std = np.std(inst_mfccs, axis=1)
                inst_mfcc_vector = np.concatenate([inst_mfcc_mean, inst_mfcc_std])

            if inst_path:
                feature_vector = np.concatenate([vocals_mfcc_vector, inst_mfcc_vector])
            else:
                feature_vector = np.concatenate([vocals_mfcc_vector])

            features.append(feature_vector)
            if data_type == "train" or data_type == "val":
                labels.append(label_id)

    if data_type == "train" or data_type == "val":
        return features, labels  # (features, labels)
    else:
        return str(vocals_path.name).replace(".mp3", ""), features  # (file_name, features)