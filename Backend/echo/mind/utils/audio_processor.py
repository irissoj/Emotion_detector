import librosa
import numpy as np
import torch

def process_audio(audio_path):
    # Extract MFCC features
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    return torch.tensor(mfcc_processed).unsqueeze(0).float()