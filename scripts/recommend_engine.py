import os
import joblib
import numpy as np
import gdown
from .audio_features import extract_features
from .perceptual_to_knobs import perceptual_to_knobs

#google drive links
MODEL_URL = "https://drive.google.com/file/d/1IwriwUezERujaXvA9Xl4rh-W435Z4aJ3/view?usp=sharing"
SCALER_URL = "https://drive.google.com/file/d/1-kclPHu9tg8id3mgQSAI_3_C3RoIr8mF/view?usp=sharing"

MODEL_PATH = "data/perceptual_model.pkl"
SCALER_PATH = "data/perceptual_scaler.pkl"


def _download(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(out_path):
        print(f"Downloading {out_path}...")
        gdown.download(url, out_path, quiet=False)


# distortion changes
def distortion_score(rms, flatness, zcr, centroid, bandwidth):
    """
    Guitar-focused distortion detector
    Returns value in [0, 1]
    """
    score = 0.0

    if centroid > 1500:
        score += 0.35

    if bandwidth > 1800:
        score += 0.25

    if zcr > 0.06:
        score += 0.25

    if flatness > 0.01:
        score += 0.15

    return min(score, 1.0)


#safety

class ToneRecommender:
    def __init__(self):
        _download(MODEL_URL, MODEL_PATH)
        _download(SCALER_URL, SCALER_PATH)

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

    def recommend(self, audio_path):
        features = extract_features(audio_path)
        X = features.reshape(1, -1)

        rms = float(features[0])
        centroid = float(features[1])
        bandwidth = float(features[2])
        zcr = float(features[3])
        flatness = float(features[4])
        perceptual_raw = self.model.predict(
            self.scaler.transform(X)
        )[0]
        perceptual = {
            "saturation": float(perceptual_raw[0]),
            "brightness": float(perceptual_raw[1]),
            "mid_emphasis": float(perceptual_raw[2]),
            "low_end": float(perceptual_raw[3]),
        }

        dist = distortion_score(
            rms, flatness, zcr, centroid, bandwidth
        )
        if dist > 0.45:
            perceptual["saturation"] = max(
                perceptual["saturation"], 0.75
            )
        final_knobs = perceptual_to_knobs(perceptual)

        return {
            "final_knobs": final_knobs,
            "perceptual": perceptual,
            "distortion_score": dist,
        }

