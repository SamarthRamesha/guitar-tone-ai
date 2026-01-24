# Guitar Tone AI — Version 1

Guitar Tone AI is an audio analysis system that estimates realistic guitar amplifier knob settings (Gain, Bass, Mid, Treble, Presence) directly from a short guitar recording.
Instead of copying presets blindly, the system analyzes how a guitar sounds and translates that into musically meaningful amp settings.

## What This Project Does
1. Accepts guitar audio files (`.wav`, `.mp3`, `.flac`, `.ogg`)
2. Extracts perceptually relevant audio features
3. Uses machine learning to predict tone characteristics
4. Converts those characteristics into real amp knob values
5. Displays results in a clean Streamlit web interface

## Core Concept
Audio → Perception → Amp Knobs
The system does not predict amp knobs directly from raw audio.  
Instead, it mimics how guitarists think about tone:
1. Analyze *what the tone sounds like*
2. Convert that perception into how an amp would be dialed
This separation makes the system more stable and musically accurate.

## Architecture Overview
1. Audio File
2. Feature Extraction (librosa)
3. Perceptual ML Model
4. Perceptual Attributes
5. Distortion Heuristic
6. Musical Mapping Logic
Amp Knob Recommendations

## Audio Features Used
1. RMS Energy (loudness)
2. Spectral Centroid (brightness)
3. Spectral Bandwidth
4. Zero Crossing Rate (distortion/noise)
5. Spectral Flatness
6.  13 MFCC coefficients
These features were chosen because they correlate strongly with perceived guitar tone and are robust across recordings.

## Machine Learning Model
1. Model Type: RandomForestRegressor  
2. Framework: scikit-learn  
3. Training Samples: ~3000 guitar recordings  
4. **Targets Predicted:**
  a. Saturation
  b. Brightness
  c. Mid Emphasis
  d. Low-End Energy  
- **Performance:** R² ≈ 0.97  
Random Forest was chosen for:
- Strong non-linear modeling
- Stability on medium-sized datasets
- High interpretability
- No heavy tuning required

## Distortion Detection
A custom **distortion score (0–1)** is computed using:
- RMS energy
- Spectral flatness
- Zero crossing rate
This prevents unrealistic results such as:
- Clean tones receiving high gain
- Distorted tones receiving low gain

## Amp Knob Mapping
Perceptual attributes are mapped to realistic amp controls using domain-informed rules:
- Clean tones → low gain, forward mids
- Crunch tones → moderate gain, balanced EQ
- High-gain tones → strong gain ramp, controlled bass, adjusted mids
All values are clamped to realistic amp ranges (0–10).

## Frontend
- Built using **Streamlit**
- Drag-and-drop audio upload
- Audio playback
- Clear amp knob display
- Human-readable tone description
- Minimal, professional UI

## Limitations (Version 1)
- Best results with **5–30 second** guitar clips
- Designed for **single guitar tracks**, not full mixes
- Does not model specific amp brands or cabinets
- Recording quality affects accuracy

## Future Improvements
- Amp family classification (Clean / Crunch / High-Gain)
- Cabinet and microphone modeling
- Time-aware feature aggregation
- Preset export for plugins or modelers
- Optional neural network models

## Tech Stack
- Python
- librosa
- NumPy
- scikit-learn
- Streamlit

## How to Run
Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run app.py

## License
Educational and experimental use only.