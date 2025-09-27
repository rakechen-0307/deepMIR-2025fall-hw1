import librosa
import numpy as np

def augment_audio(y, sr, time_stretch=False, pitch_shift=False, noise_injection=False):
    """
    Apply data augmentation techniques to the audio signal.
    """
    if time_stretch:
        rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y=y, rate=rate)

    if pitch_shift:
        n_steps = np.random.randint(-2, 3)
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

    if noise_injection:
        noise = np.random.normal(0, 0.002, len(y))
        y = y + noise

    return y