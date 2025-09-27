import joblib
import librosa
import numpy as np
from contextlib import contextmanager

@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()

def get_intervals(
    y, sr, split_audio=False, silent_threshold=30, 
    min_seg=10.0, max_seg=15.0, max_silence=10.0
):
    """
    Split audio into non-silent intervals based on silence detection.
    """
    if split_audio:
        non_silences = librosa.effects.split(y=y, top_db=silent_threshold)
        silences = []
        for j in range(len(non_silences) - 1):
            silence_start = non_silences[j][1]
            silence_end = non_silences[j + 1][0]
            silences.append(silence_end - silence_start)
        intervals = []
        curr_start = non_silences[0][0]
        j = 0
        while j < len(silences):
            curr_end = non_silences[j][1]
            if (curr_end - curr_start) / sr >= max_seg:
                if (non_silences[j][0] == curr_start):
                    # split this segment into pieces of max_seg
                    num_full_segs = int((curr_end - curr_start) / sr // max_seg)
                    for k in range(num_full_segs):
                        intervals.append([curr_start + int(k * max_seg * sr), curr_start + int((k + 1) * max_seg * sr)])
                    if (curr_end - (curr_start + int(num_full_segs * max_seg * sr))) / sr >= min_seg:
                        intervals.append([curr_start + int(num_full_segs * max_seg * sr), curr_end])
                    curr_start = non_silences[j + 1][0]
                    j += 1
                else:
                    if (non_silences[j - 1][1] - curr_start) / sr >= min_seg:
                        intervals.append([curr_start, non_silences[j - 1][1]])
                    curr_start = non_silences[j][0]
            elif silences[j] / sr >= max_silence:
                if (curr_end - curr_start) / sr >= min_seg:
                    intervals.append([curr_start, curr_end])
                curr_start = non_silences[j + 1][0]
                j += 1
            else:
                j += 1
        # Check for the last segment
        if (non_silences[-1][1] - curr_start) / sr >= max_seg:
            num_full_segs = int((non_silences[-1][1] - curr_start) / sr // max_seg)
            for k in range(num_full_segs):
                intervals.append([curr_start + int(k * max_seg * sr), curr_start + int((k + 1) * max_seg * sr)])
        elif (non_silences[-1][0] - curr_start) / sr >= min_seg:
            intervals.append([curr_start, non_silences[-1][0]])

        if len(intervals) == 0:
            # Fall back if no valid segments found
            num_full_segs = int(len(y) / sr // max_seg)
            if num_full_segs >= 1:
                intervals = []
                for k in range(num_full_segs):
                    intervals.append([int(k * max_seg * sr), int((k + 1) * max_seg * sr)])
                if len(y) - int(num_full_segs * max_seg * sr) >= min_seg * sr:
                    intervals.append([int(num_full_segs * max_seg * sr), len(y)])
                intervals = np.array(intervals)
            else:
                intervals = np.array([[0, len(y)]])
        else:
            intervals = np.array(intervals)
    else:
        num_full_segs = int(len(y) / sr // max_seg)
        if num_full_segs >= 1:
            intervals = []
            for k in range(num_full_segs):
                intervals.append([int(k * max_seg * sr), int((k + 1) * max_seg * sr)])
            if len(y) - int(num_full_segs * max_seg * sr) >= min_seg * sr:
                intervals.append([int(num_full_segs * max_seg * sr), len(y)])
            intervals = np.array(intervals)
        else:
            intervals = np.array([[0, len(y)]])

    return intervals