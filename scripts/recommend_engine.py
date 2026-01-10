# scripts/recommend_engine.py

import joblib
import numpy as np

from .audio_features import extract_features
from .perceptual_to_knobs import perceptual_to_knobs


def distortion_score(rms, flatness, zcr, centroid, bandwidth):
    """
    Guitar-focused distortion detector
    Returns value in [0, 1]
    """
    score = 0.0

    #guitar distortion is not noise, it's harmonic density
    if centroid > 1500:
        score += 0.35

    if bandwidth > 1800:
        score += 0.25

    if zcr > 0.06:
        score += 0.25

    #flatness is weak for guitar, small weight
    if flatness > 0.01:
        score += 0.15

    return min(score, 1.0)


class ToneRecommender:
    def __init__(self):
        self.model = joblib.load("data/perceptual_model.pkl")
        self.scaler = joblib.load("data/perceptual_scaler.pkl")

    def recommend(self, audio_path):
        #feature extraction
        features = extract_features(audio_path)
        X = features.reshape(1, -1)

        rms = float(features[0])
        centroid = float(features[1])
        bandwidth = float(features[2])
        zcr = float(features[3])
        flatness = float(features[4])

        #perceptual prediction
        perceptual_raw = self.model.predict(
            self.scaler.transform(X)
        )[0]

        perceptual = {
            "saturation": float(perceptual_raw[0]),
            "brightness": float(perceptual_raw[1]),
            "mid_emphasis": float(perceptual_raw[2]),
            "low_end": float(perceptual_raw[3]),
        }

        #distortion override
        dist = distortion_score(
            rms, flatness, zcr, centroid, bandwidth
        )

        # HARD FORCE for distorted guitar
        if dist > 0.45:
            perceptual["saturation"] = max(
                perceptual["saturation"], 0.75
            )

        #knob mapping
        final_knobs = perceptual_to_knobs(perceptual)

        return {
            "final_knobs": final_knobs,
            "perceptual": perceptual,
            "distortion_score": dist,
        }