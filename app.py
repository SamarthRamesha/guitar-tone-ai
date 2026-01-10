import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import tempfile

from scripts.recommend_engine import ToneRecommender

# -------------------------------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT CALL)
# -------------------------------------------------

st.set_page_config(
    page_title="Guitar Tone AI",
    page_icon="🎸",
    layout="centered"
)

# -------------------------------------------------
# LOAD MODEL (CACHED)
# -------------------------------------------------

@st.cache_resource
def load_engine():
    return ToneRecommender()

engine = load_engine()

# -------------------------------------------------
# STYLES
# -------------------------------------------------

st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.hero {
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.sub {
    color: #9da7b3;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.panel {
    background: #12161c;
    padding: 1.1rem 1.3rem;
    border-radius: 10px;
    border: 1px solid #1f2630;
    margin-bottom: 1.2rem;
}

.small {
    color: #8b949e;
    font-size: 0.85rem;
}

.metric-label {
    color: #9da7b3;
    font-size: 0.8rem;
}

hr {
    border: none;
    border-top: 1px solid #1f2630;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown("<div class='hero'>🎸 Guitar Tone AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub'>Perceptual amplifier tone estimation powered by machine learning</div>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# UPLOAD
# -------------------------------------------------

uploaded = st.file_uploader(
    "Upload guitar audio",
    type=["wav", "mp3", "flac", "ogg"],
    help="Short, focused guitar recordings produce the most accurate results."
)

# -------------------------------------------------
# CONTEXT PANEL
# -------------------------------------------------

st.markdown("""
<div class="panel">
<strong>How interpretation works</strong><br><br>

The system analyzes spectral balance, harmonic density, and dynamic behavior
to infer amplifier-style controls rather than recreating effects chains.

<hr>

<div class="small">
• Best with consistent tones<br>
• Optimized for guitar-forward audio<br>
• Interprets tone character, not performance technique
</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PROCESS AUDIO
# -------------------------------------------------

if uploaded is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded.read())
            audio_path = tmp.name

        st.audio(uploaded)

        result = engine.recommend(audio_path)

        os.remove(audio_path)

        knobs = result["final_knobs"]
        perceptual = result["perceptual"]
        distortion = result["distortion_score"]
        confidence = result.get("confidence", 0.75)

        # -------------------------------------------------
        # AMP SETTINGS
        # -------------------------------------------------

        st.markdown("## Recommended Amp Controls")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Gain", f"{knobs['gain']:.2f}")
        c2.metric("Bass", f"{knobs['bass']:.2f}")
        c3.metric("Mid", f"{knobs['mid']:.2f}")
        c4.metric("Treble", f"{knobs['treble']:.2f}")
        c5.metric("Presence", f"{knobs['presence']:.2f}")

        # -------------------------------------------------
        # TONE PROFILE
        # -------------------------------------------------

        with st.expander("Tone profile"):
            def bar(label, value, hint):
                st.markdown(f"**{label}**")
                st.progress(min(max(value, 0.0), 1.0))
                st.caption(hint)

            bar("Saturation", perceptual["saturation"],
                "Harmonic density and drive intensity.")

            bar("Brightness", perceptual["brightness"],
                "High-frequency energy and bite.")

            bar("Mid Emphasis", perceptual["mid_emphasis"],
                "Midrange forwardness or scoop.")

            bar("Low-End Weight", perceptual["low_end"],
                "Bass fullness and stability.")

            st.markdown("---")
            st.markdown("**Distortion Intensity**")
            st.progress(distortion)

        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        st.markdown("## Confidence")
        st.progress(confidence)
        st.caption(
            "Confidence reflects similarity to learned tonal patterns."
        )

    except Exception as e:
        st.error("An error occurred while processing the audio.")
        st.exception(e)
