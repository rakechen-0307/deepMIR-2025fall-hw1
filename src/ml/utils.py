import librosa
import numpy as np

from ..mapping import artist_code_map

def extract_features(
    vocals_path, inst_path=None, sr=16000, split_audio=False,
    silent_threshold=50, min_seg=5.0, is_training=True
):
    """
    Extract audio features from a given file. 
    """
    y_vocals, _ = librosa.load(vocals_path, sr=sr)
    y_vocals = librosa.util.normalize(y_vocals)  # Normalize audio
    if inst_path:
        y_inst, _ = librosa.load(inst_path, sr=sr)
        y_inst = librosa.util.normalize(y_inst)  # Normalize audio

    if is_training:
        label = str(vocals_path).split("/")[-3]
        label_id = artist_code_map[label]

    if split_audio:
        intervals = librosa.effects.split(y_vocals, top_db=silent_threshold)
        # Filter out short segments
        intervals = [interval for interval in intervals if (interval[1] - interval[0]) / sr >= min_seg]
        if len(intervals) == 0:
            intervals = np.array([[0, len(y_vocals)]])
    else:
        intervals = np.array([[0, len(y_vocals)]])

    if is_training:
        features, labels = [], []
    else:   
        features = []

    for interval in intervals:
        start, end = interval
        vocals_segment = y_vocals[start:end]
        if inst_path:
            inst_segment = y_inst[start:end]
            
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
        if is_training:
            labels.append(label_id)

    if is_training:
        return features, labels  # (features, labels)
    else:
        return str(vocals_path.name).replace(".mp3", ""), features  # (file_name, features)