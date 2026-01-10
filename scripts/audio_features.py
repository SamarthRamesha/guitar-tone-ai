import librosa
import numpy as np

TARGET_SR = 44100

def extract_features(audio_path):
    # Load ANY audio format
    y, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True,
        res_type="kaiser_best"
    )

    # Safety: trim silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # Normalize loudness (important for RMS consistency)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # ---- Feature extraction ----
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    features = np.hstack([
        rms,
        centroid,
        bandwidth,
        zcr,
        flatness,
        mfcc_mean
    ])

    return features