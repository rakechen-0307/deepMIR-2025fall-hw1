import librosa
import numpy as np

from ..mapping import artist_code_map

def extract_features(
    file_path, sr=16000, n_mfcc=13, len_segment=0,
    return_label=True
):
    """
    Extract audio features from a given file. 
    """
    y, _ = librosa.load(file_path, sr=sr)
    if return_label:
        label = str(file_path).split("/")[-3]
        label_id = artist_code_map[label]

    segment_samples = len_segment * sr if len_segment > 0 else len(y)
    if return_label:
        features, labels = [], []
    else:   
        features = []

    for start in range(0, len(y), segment_samples):
        end = start + segment_samples
        segment = y[start:end]

        if len(segment) < segment_samples:  # drop too short last piece
            continue

        # ========== MFCCs ==========
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_vector = np.concatenate([mfcc_mean, mfcc_std])

        # TODO: other features

        feature_vector = np.concatenate([mfcc_vector])

        features.append(feature_vector)
        if return_label:
            labels.append(label_id)

    if return_label:
        return features, labels
    else:
        return features