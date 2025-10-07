import librosa
import numpy as np
from random import random

from ..commons import get_intervals
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
    data_type, vocals_path, inst_path=None, sr=16000, split_audio=False,
    silent_threshold=30, min_seg=10.0, max_seg=15.0, max_silence=10.0, 
    num_augments=0, time_stretch_ratio=0.5, pitch_shift_ratio=0.5, 
    noise_injection_ratio=0.5, used_features=["mfcc", "chroma", "beat"]
):
    """
    Extract audio features from a given file. 
    """
    assert data_type in ["train", "val", "test"], "data_type must be 'train', 'val', or 'test'"
    assert all(f in ["mfcc", "chroma", "beat"] for f in used_features), "used_features must be a list containing any of 'mfcc', 'chroma', 'beat'"

    y_vocals, _ = librosa.load(path=vocals_path, sr=sr)
    y_vocals = librosa.util.normalize(S=y_vocals)  # Normalize audio
    if inst_path is not None:
        y_inst, _ = librosa.load(path=inst_path, sr=sr)
        y_inst = librosa.util.normalize(S=y_inst)  # Normalize audio

    if data_type == "train" or data_type == "val":
        label = str(vocals_path).split("/")[-3]
        label_id = artist_code_map[label]
        features, labels = [], []
    else:
        features = []

    if data_type == "train":
        num_samples = num_augments + 1  # original + augmentations
    else:
        num_samples = 1  # only original

    for i in range(num_samples):
        intervals = get_intervals(
            y=y_vocals, sr=sr, split_audio=split_audio, silent_threshold=silent_threshold,
            min_seg=min_seg, max_seg=max_seg, max_silence=max_silence
        )

        for interval in intervals:
            start, end = interval
            vocals_segment = y_vocals[start:end]
            if inst_path is not None:
                inst_segment = y_inst[start:end]

            if i != 0:
                time_stretch = True if random() < time_stretch_ratio else False
                pitch_shift = True if random() < pitch_shift_ratio else False
                noise_injection = True if random() < noise_injection_ratio else False
                if inst_path is not None:
                    vocals_segment, inst_segment = augment_audio(
                        y_vocals=vocals_segment, y_inst=inst_segment, sr=sr, 
                        time_stretch=time_stretch, pitch_shift=pitch_shift, 
                        noise_injection=noise_injection
                    )
                else:
                    vocals_segment = augment_audio(
                        y_vocals=vocals_segment, y_inst=None, sr=sr, 
                        time_stretch=time_stretch, pitch_shift=pitch_shift, 
                        noise_injection=noise_injection
                    )

            # ========== Vocals MFCCs ==========
            if "mfcc" in used_features:
                vocals_mfccs = librosa.feature.mfcc(
                    y=vocals_segment, sr=sr, n_mfcc=13, n_fft=512, hop_length=160,
                    fmin=65, fmax=8000, n_mels=128
                )
                vocals_mfcc_mean = np.mean(vocals_mfccs, axis=1)
                vocals_mfcc_std = np.std(vocals_mfccs, axis=1)
                vocals_mfcc_vector = np.concatenate([vocals_mfcc_mean, vocals_mfcc_std])
            else:
                vocals_mfcc_vector = np.array([])

            # ========== Vocals Chroma ==========
            if "chroma" in used_features:
                vocals_chroma = librosa.feature.chroma_cqt(
                    y=vocals_segment, sr=sr, n_chroma=12, n_octaves=7, 
                    hop_length=512, fmin=librosa.note_to_hz('C1')
                )
                vocals_chroma_mean = np.mean(vocals_chroma, axis=1)
                vocals_chroma_std = np.std(vocals_chroma, axis=1)
                vocals_chroma_vector = np.concatenate([vocals_chroma_mean, vocals_chroma_std])
            else:
                vocals_chroma_vector = np.array([])

            # ========== Vocals Beat ==========
            if "beat" in used_features:
                vocals_plp = librosa.beat.plp(y=vocals_segment, sr=sr, win_length=256, hop_length=160)
                vocals_plp_mean = np.mean(vocals_plp)
                vocals_plp_std = np.std(vocals_plp)
                tempo = librosa.feature.rhythm.tempo(onset_envelope=vocals_plp, sr=sr, hop_length=160)
                vocals_tempo_mean = np.mean(tempo)
                vocals_tempo_std = np.std(tempo)
                vocals_beat_vector = np.array([vocals_plp_mean, vocals_plp_std, vocals_tempo_mean, vocals_tempo_std])
            else:
                vocals_beat_vector = np.array([])

            if inst_path is not None:
                # ========== Instrumental MFCCs ==========
                if "mfcc" in used_features:
                    inst_mfccs = librosa.feature.mfcc(
                        y=inst_segment, sr=sr, n_mfcc=5, n_fft=512, hop_length=160,
                        fmin=20, fmax=8000, n_mels=128
                    )
                    inst_mfcc_mean = np.mean(inst_mfccs, axis=1)
                    inst_mfcc_std = np.std(inst_mfccs, axis=1)
                    inst_mfcc_vector = np.concatenate([inst_mfcc_mean, inst_mfcc_std])
                else:
                    inst_mfcc_vector = np.array([])

                # ========== Instrumental Chroma ==========
                if "chroma" in used_features:
                    inst_chroma = librosa.feature.chroma_cqt(
                        y=inst_segment, sr=sr, n_chroma=12, n_octaves=7,
                        hop_length=512, fmin=librosa.note_to_hz('C1')
                    )
                    inst_chroma_mean = np.mean(inst_chroma, axis=1)
                    inst_chroma_std = np.std(inst_chroma, axis=1)
                    inst_chroma_vector = np.concatenate([inst_chroma_mean, inst_chroma_std])
                else:
                    inst_chroma_vector = np.array([])

                # ========== Instrumental Beat ==========
                if "beat" in used_features:
                    inst_plp = librosa.beat.plp(y=inst_segment, sr=sr, win_length=512, hop_length=160)
                    inst_plp_mean = np.mean(inst_plp)
                    inst_plp_std = np.std(inst_plp)
                    tempo = librosa.feature.rhythm.tempo(onset_envelope=inst_plp, sr=sr, hop_length=160)
                    inst_tempo_mean = np.mean(tempo)
                    inst_tempo_std = np.std(tempo)
                    inst_beat_vector = np.array([inst_plp_mean, inst_plp_std, inst_tempo_mean, inst_tempo_std])
                else:
                    inst_beat_vector = np.array([])

            if inst_path is not None:
                feature_vector = np.concatenate([vocals_mfcc_vector, vocals_chroma_vector, vocals_beat_vector, inst_mfcc_vector, inst_chroma_vector, inst_beat_vector])
            else:
                feature_vector = np.concatenate([vocals_mfcc_vector, vocals_chroma_vector, vocals_beat_vector])

            features.append(feature_vector)
            if data_type == "train" or data_type == "val":
                labels.append(label_id)

    if data_type == "train" or data_type == "val":
        return features, labels  # (features, labels)
    else:
        return str(vocals_path.name).replace(".mp3", ""), features  # (file_name, features)