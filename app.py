"""
EcoGuard | Real-Time Acoustic Threat Detection System
=====================================================
Single-file Streamlit app for the Forest Threat Sound Detection project.

Run:   streamlit run app.py
Needs: models/*.joblib and models/*_config.json (or the same files next to app.py)

Notes on the trained models (taken from the project notebook):
  * Developer A models (LightGBM, Random Forest, Logistic Regression) are pickled
    `forest_model_api.ForestSoundModel` objects. Their feature extractor uses
    frame_length=N_FFT for ZCR and a leading 0 for spectral flux, and the class
    index order is ALPHABETICAL (chainsaw_engine ... unknown).
  * Developer B models (CatBoost, SVM-RBF, XGBoost) are plain estimators trained on a
    slightly different extractor (default ZCR frame length, no leading flux zero,
    1-based feature names). Their labels come from sklearn LabelEncoder, i.e. ALSO
    alphabetical - NOT the order printed in their *_config.json `class_names`.
  Both extractors are reproduced here exactly, so each model sees the features it
  was trained on.
"""

import hashlib
import io
import json
import os
import sys
import tempfile
import types
import warnings

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONSTANTS
# ==============================================================================
APP_TITLE = "EcoGuard | Real-Time Acoustic Threat Detection System"
SR = 16_000
WIN = 48_000  # 3 s analysis window, as used in training
N_FFT = 512
HOP = 160
N_MFCC = 20
CLASSES = ["chainsaw_engine", "fire", "gunshot", "human_voice", "natural", "rain_thunder", "unknown"]
CLASSES_B_ORDER = sorted(CLASSES)  # sklearn LabelEncoder order used by Developer B
DEFAULT_THREATS = ["chainsaw_engine", "fire", "gunshot"]

CLASS_META = {
    "chainsaw_engine": ("🪚", "Chainsaw / Engine", "Possible illegal logging activity"),
    "fire": ("🔥", "Fire", "Crackling consistent with wildfire"),
    "gunshot": ("🔫", "Gunshot", "Possible poaching or armed intrusion"),
    "human_voice": ("🗣️", "Human Voice", "Human presence in the area"),
    "natural": ("🌿", "Natural Forest Sounds", "Birds, wind and ambient forest noise"),
    "rain_thunder": ("🌧️", "Rain / Thunder", "Weather sounds"),
    "unknown": ("❔", "Unknown / Other", "Unclassified background"),
}

MODEL_DISPLAY = {
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "svm_rbf": "SVM (RBF)",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
}
MODEL_ORDER = list(MODEL_DISPLAY.keys())

NEON, DEEP, DARKEST, PANEL = "#00FF66", "#0D3B1E", "#051C0D", "#122619"
RED, ORANGE, AMBER = "#FF2D55", "#FF8A00", "#FFD60A"

# The module that the Developer A pickles reference (verbatim from the notebook).
FOREST_MODEL_API_SRC = r'''
import numpy as np
import pandas as pd
import librosa

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 48_000
FRAME_LENGTH_MS = 32
HOP_LENGTH_MS = 10
N_FFT = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)
HOP_LENGTH = int(SAMPLE_RATE * HOP_LENGTH_MS / 1000)
N_MFCC = 20


def extract_frame_level_features(wav, sr=SAMPLE_RATE):
    frame_feats = {}
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    for i in range(N_MFCC):
        frame_feats[f"mfcc_{i}"] = mfcc[i]
        frame_feats[f"mfcc_delta_{i}"] = delta_mfcc[i]
        frame_feats[f"mfcc_delta2_{i}"] = delta2_mfcc[i]
    frame_feats["spectral_centroid"] = librosa.feature.spectral_centroid(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    frame_feats["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    frame_feats["spectral_rolloff"] = librosa.feature.spectral_rolloff(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    contrast = librosa.feature.spectral_contrast(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(contrast.shape[0]):
        frame_feats[f"spectral_contrast_{i}"] = contrast[i]
    frame_feats["zcr"] = librosa.feature.zero_crossing_rate(wav, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    frame_feats["rms"] = librosa.feature.rms(y=wav, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    stft_mag = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH))
    flux = np.sqrt(np.sum(np.diff(stft_mag, axis=1) ** 2, axis=0))
    frame_feats["spectral_flux"] = np.concatenate([[0.0], flux])
    return frame_feats


def aggregate_features(frame_feats):
    agg = {}
    for name, series in frame_feats.items():
        series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
        agg[f"{name}_mean"] = float(np.mean(series))
        agg[f"{name}_std"] = float(np.std(series))
        agg[f"{name}_min"] = float(np.min(series))
        agg[f"{name}_max"] = float(np.max(series))
    return agg


class ForestSoundModel:
    """Model API contract: predict(window) -> dict[class, prob]. Carries its own config."""

    def __init__(self, pipeline, class_names, feature_cols, config):
        self.pipeline = pipeline
        self.class_names = list(class_names)     # index -> class name
        self.feature_cols = list(feature_cols)   # exact training column order
        self.config = dict(config)

    def predict(self, window):
        w = np.asarray(window, dtype=np.float32).ravel()
        if len(w) < WINDOW_SAMPLES:
            w = np.pad(w, (0, WINDOW_SAMPLES - len(w)))
        elif len(w) > WINDOW_SAMPLES:
            w = w[:WINDOW_SAMPLES]
        feats = aggregate_features(extract_frame_level_features(w, SAMPLE_RATE))
        x = pd.DataFrame([feats]).reindex(columns=self.feature_cols)
        proba = self.pipeline.predict_proba(x)[0]
        out = {c: 0.0 for c in self.class_names}
        for cls_idx, p in zip(self.pipeline.classes_, proba):
            out[self.class_names[int(cls_idx)]] = float(p)
        return out
'''


def _install_forest_model_api():
    """Register `forest_model_api` in sys.modules so joblib can unpickle Developer A models."""
    if "forest_model_api" in sys.modules and hasattr(sys.modules["forest_model_api"], "ForestSoundModel"):
        return sys.modules["forest_model_api"]
    mod = types.ModuleType("forest_model_api")
    exec(FOREST_MODEL_API_SRC, mod.__dict__)
    sys.modules["forest_model_api"] = mod
    return mod


API = _install_forest_model_api()

# ==============================================================================
# 1. PAGE CONFIG + THEME
# ==============================================================================
st.set_page_config(page_title="EcoGuard", page_icon="🌲", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600&display=swap');
:root{--neon:#00FF66;--deep:#0D3B1E;--dark:#051C0D;--panel:#122619;--red:#FF2D55;--orange:#FF8A00;--text:#C9F7D9;}
html, body, [class*="css"], .stMarkdown, p, label, span{font-family:'Rajdhani',sans-serif;}
.stApp{
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba(0,255,102,0.12), transparent 60%),
    radial-gradient(900px 500px at 100% 10%, rgba(13,59,30,0.9), transparent 60%),
    linear-gradient(160deg,#030f07 0%,#051C0D 45%,#0a2414 100%);
  color:var(--text);
}
header[data-testid="stHeader"]{background:transparent;}
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#041509 0%,#0a2112 100%);
  border-right:1px solid rgba(0,255,102,0.25);
  box-shadow:4px 0 24px rgba(0,255,102,0.08);
}
h1,h2,h3,h4{font-family:'Orbitron',sans-serif !important;color:var(--neon) !important;letter-spacing:.5px;}
.eg-hero{
  padding:26px 30px;margin-bottom:18px;border-radius:18px;
  background:linear-gradient(135deg,rgba(18,38,25,.85),rgba(5,28,13,.85));
  border:1px solid rgba(0,255,102,.45);
  box-shadow:0 0 28px rgba(0,255,102,.18), inset 0 0 40px rgba(0,255,102,.05);
  backdrop-filter:blur(8px);
}
.eg-hero h1{margin:0;font-size:2.0rem;text-shadow:0 0 14px rgba(0,255,102,.7);}
.eg-hero p{margin:6px 0 0 0;color:#8fe6b0;font-size:1.05rem;}
.eg-card{
  padding:18px 20px;margin:8px 0 14px 0;border-radius:16px;
  background:rgba(18,38,25,.62);border:1px solid rgba(0,255,102,.28);
  box-shadow:0 0 18px rgba(0,255,102,.10);backdrop-filter:blur(10px);
}
.eg-kpi{text-align:center;padding:14px 8px;border-radius:14px;background:rgba(18,38,25,.7);
  border:1px solid rgba(0,255,102,.3);box-shadow:0 0 14px rgba(0,255,102,.12);}
.eg-kpi .v{font-family:'Orbitron',sans-serif;font-size:1.5rem;color:var(--neon);text-shadow:0 0 10px rgba(0,255,102,.6);}
.eg-kpi .l{font-size:.85rem;color:#8fe6b0;text-transform:uppercase;letter-spacing:1.5px;}
.alert{padding:24px 28px;border-radius:18px;margin:10px 0 16px 0;}
.alert .badge{display:inline-block;padding:5px 14px;border-radius:999px;font-family:'Orbitron',sans-serif;
  font-size:.78rem;letter-spacing:1.5px;font-weight:700;margin-bottom:10px;}
.alert .cls{font-family:'Orbitron',sans-serif;font-size:1.8rem;font-weight:900;margin:2px 0;}
.alert .pct{font-family:'Orbitron',sans-serif;font-size:3rem;font-weight:900;line-height:1.1;}
.alert .sub{font-size:1.05rem;opacity:.9;margin-top:4px;}
.alert-threat{
  background:linear-gradient(135deg,rgba(80,0,16,.85),rgba(40,6,10,.9));
  border:2px solid var(--red);color:#ffd9e0;animation:pulseRed 1.6s ease-in-out infinite;
}
.alert-threat .badge{background:var(--red);color:#fff;box-shadow:0 0 16px var(--red);}
.alert-threat .cls,.alert-threat .pct{color:#ff5a7a;text-shadow:0 0 18px rgba(255,45,85,.9);}
@keyframes pulseRed{0%,100%{box-shadow:0 0 18px rgba(255,45,85,.45),inset 0 0 22px rgba(255,45,85,.12);}
  50%{box-shadow:0 0 44px rgba(255,45,85,.95),inset 0 0 36px rgba(255,45,85,.25);}}
.alert-safe{
  background:linear-gradient(135deg,rgba(5,60,28,.8),rgba(5,28,13,.9));
  border:2px solid var(--neon);color:#d6ffe6;animation:pulseGreen 2.6s ease-in-out infinite;
}
.alert-safe .badge{background:var(--neon);color:#021207;box-shadow:0 0 16px var(--neon);}
.alert-safe .cls,.alert-safe .pct{color:var(--neon);text-shadow:0 0 18px rgba(0,255,102,.85);}
@keyframes pulseGreen{0%,100%{box-shadow:0 0 16px rgba(0,255,102,.35);}50%{box-shadow:0 0 34px rgba(0,255,102,.7);}}
.pill{display:inline-block;padding:3px 12px;margin:2px 4px 2px 0;border-radius:999px;font-size:.85rem;
  border:1px solid rgba(0,255,102,.5);color:var(--neon);background:rgba(0,255,102,.08);}
.pill-red{border-color:var(--red);color:#ff7d94;background:rgba(255,45,85,.1);}
.stTabs [data-baseweb="tab-list"]{gap:6px;}
.stTabs [data-baseweb="tab"]{background:rgba(18,38,25,.7);border:1px solid rgba(0,255,102,.25);
  border-radius:10px 10px 0 0;padding:8px 18px;color:#9be8b8;font-family:'Orbitron',sans-serif;font-size:.8rem;}
.stTabs [aria-selected="true"]{background:rgba(0,255,102,.15) !important;color:var(--neon) !important;
  border-color:var(--neon) !important;box-shadow:0 0 14px rgba(0,255,102,.35);}
.stButton>button,.stDownloadButton>button{
  background:linear-gradient(135deg,#0D3B1E,#0b5a2b);color:var(--neon);border:1px solid var(--neon);
  border-radius:12px;font-family:'Orbitron',sans-serif;letter-spacing:1px;
  box-shadow:0 0 14px rgba(0,255,102,.35);transition:all .2s;}
.stButton>button:hover,.stDownloadButton>button:hover{box-shadow:0 0 28px rgba(0,255,102,.8);
  color:#fff;border-color:#7dffb0;transform:translateY(-1px);}
div[data-testid="stFileUploader"] section{background:rgba(18,38,25,.6);border:1px dashed rgba(0,255,102,.55);border-radius:14px;}
div[data-testid="stDataFrame"]{border:1px solid rgba(0,255,102,.3);border-radius:12px;box-shadow:0 0 14px rgba(0,255,102,.1);}
hr{border-color:rgba(0,255,102,.2) !important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. SMALL UI HELPERS
# ==============================================================================
def show_plot(fig, key=None):
    """st.plotly_chart across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df, key=None, **kwargs):
    """st.dataframe across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, key=key, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def style_fig(fig, height=320, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,28,13,0.55)",
        font=dict(color="#B8F5CF", family="Rajdhani, sans-serif", size=13),
        margin=dict(l=48, r=20, t=48 if title else 20, b=40),
        height=height,
        title=dict(text=title, font=dict(color=NEON, family="Orbitron, sans-serif", size=15)) if title else None,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(0,255,102,0.10)", zerolinecolor="rgba(0,255,102,0.2)")
    fig.update_yaxes(gridcolor="rgba(0,255,102,0.10)", zerolinecolor="rgba(0,255,102,0.2)")
    return fig


def pretty(c):
    return CLASS_META.get(c, ("", c.replace("_", " ").title(), ""))[1]


def emoji(c):
    return CLASS_META.get(c, ("🔊", "", ""))[0]


def kpi(col, value, label):
    col.markdown(f'<div class="eg-kpi"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


# ==============================================================================
# 3. FEATURE EXTRACTION  (two extractors, exactly as in the notebook)
# ==============================================================================
def fit_window(w):
    w = np.asarray(w, dtype=np.float32).ravel()
    if len(w) < WIN:
        return np.pad(w, (0, WIN - len(w)))
    return w[:WIN]


def features_dev_a(w):
    """Developer A extractor (forest_model_api.py): used by LightGBM / RF / LogReg."""
    return API.aggregate_features(API.extract_frame_level_features(w, SR))


def frame_features_dev_b(wav, sr=SR):
    """Developer B extractor (notebook 'CLASSICAL ML - SECTION 1'): CatBoost / SVM / XGBoost."""
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    centroid = librosa.feature.spectral_centroid(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)
    bandwidth = librosa.feature.spectral_bandwidth(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)
    rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)
    contrast = librosa.feature.spectral_contrast(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)
    zcr = librosa.feature.zero_crossing_rate(wav, hop_length=HOP)  # default frame_length (as trained)
    rms = librosa.feature.rms(y=wav, frame_length=N_FFT, hop_length=HOP)
    stft = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP))
    flux = np.asarray(np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0)))
    if len(flux) == 0:
        flux = np.zeros(1)

    ff = {}
    for i in range(N_MFCC):
        ff[f"mfcc_{i + 1}"] = mfcc[i]
    for i in range(N_MFCC):
        ff[f"delta_mfcc_{i + 1}"] = delta_mfcc[i]
    for i in range(N_MFCC):
        ff[f"delta2_mfcc_{i + 1}"] = delta2_mfcc[i]
    ff["spectral_centroid"] = centroid[0]
    ff["spectral_bandwidth"] = bandwidth[0]
    ff["spectral_rolloff"] = rolloff[0]
    for i in range(contrast.shape[0]):
        ff[f"spectral_contrast_{i + 1}"] = contrast[i]
    ff["zero_crossing_rate"] = zcr[0]
    ff["rms"] = rms[0]
    ff["spectral_flux"] = flux
    return ff


def features_dev_b(w):
    out = {}
    for name, values in frame_features_dev_b(w).items():
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            v = np.array([0.0])
        out[f"{name}_mean"] = np.mean(v)
        out[f"{name}_std"] = np.std(v)
        out[f"{name}_min"] = np.min(v)
        out[f"{name}_max"] = np.max(v)
    return out


def make_windows(y, max_windows):
    """Split a clip into consecutive 3 s windows (last partial window kept if >= 1 s)."""
    n = len(y)
    if n <= WIN:
        starts = [0]
    else:
        starts = list(range(0, n - WIN + 1, WIN))
        if n - (starts[-1] + WIN) >= SR:
            starts.append(starts[-1] + WIN)
    starts = starts[:max_windows]
    return [s / SR for s in starts], [fit_window(y[s:s + WIN]) for s in starts]


@st.cache_data(show_spinner=False, max_entries=6)
def compute_features(y, max_windows):
    times, wins = make_windows(y, max_windows)
    rows_a = [features_dev_a(w) for w in wins]
    rows_b = [features_dev_b(w) for w in wins]
    df_a = pd.DataFrame(rows_a).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df_b = pd.DataFrame(rows_b).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return times, wins, df_a, df_b


# ==============================================================================
# 4. AUDIO LOADING, SAMPLES, VISUALS
# ==============================================================================
@st.cache_data(show_spinner=False, max_entries=6)
def load_audio(data: bytes, filename: str):
    try:
        y, _ = librosa.load(io.BytesIO(data), sr=SR, mono=True)
    except Exception:
        suffix = os.path.splitext(filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            path = f.name
        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
        finally:
            os.unlink(path)
    return y.astype(np.float32)


def _to_wav_bytes(y):
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def synth_samples():
    """Synthetic test signals - ONLY for checking the UI/pipeline, not real recordings."""
    rng = np.random.default_rng(7)
    t = np.arange(WIN) / SR
    rain = rng.normal(0, 0.15, WIN) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.7 * t))
    engine = sum(np.sin(2 * np.pi * 90 * k * t) / k for k in range(1, 12)) * (0.6 + 0.4 * np.sin(2 * np.pi * 9 * t))
    engine = 0.25 * engine + rng.normal(0, 0.03, WIN)
    burst = np.zeros(WIN)
    for pos in (int(0.4 * SR), int(1.5 * SR), int(2.3 * SR)):
        n = int(0.25 * SR)
        burst[pos:pos + n] += rng.normal(0, 1, n) * np.exp(-np.arange(n) / (0.03 * SR))
    burst = 0.8 * burst / (np.abs(burst).max() + 1e-9) + rng.normal(0, 0.005, WIN)
    chirps = np.zeros(WIN)
    for start in (0.2, 1.0, 1.9):
        m = (t >= start) & (t < start + 0.35)
        f = 2500 + 2500 * (t[m] - start) / 0.35
        chirps[m] = 0.35 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.hanning(m.sum())
    chirps += rng.normal(0, 0.01, WIN)
    return {
        "Synthetic · rain-like noise": rain,
        "Synthetic · engine-like harmonics": engine,
        "Synthetic · impulsive bursts": burst,
        "Synthetic · bird-like chirps": chirps,
    }


def waveform_fig(y, window_times=None):
    n_bins = 1500
    step = max(1, len(y) // n_bins)
    trim = len(y) // step * step
    chunks = y[:trim].reshape(-1, step)
    t = (np.arange(chunks.shape[0]) * step) / SR
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=chunks.max(1), mode="lines", line=dict(color=NEON, width=1), name="peak+",
                             fill=None, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=chunks.min(1), mode="lines", line=dict(color=NEON, width=1), name="peak-",
                             fill="tonexty", fillcolor="rgba(0,255,102,0.35)", hoverinfo="skip"))
    if window_times and len(window_times) > 1:
        for wt in window_times[1:]:
            fig.add_vline(x=wt, line=dict(color="rgba(0,255,102,0.35)", dash="dot", width=1))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Amplitude")
    return style_fig(fig, 260, "Waveform  (dotted lines = 3 s analysis windows)")


def mel_db(y):
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=256, n_mels=96, fmax=SR // 2)
    return librosa.power_to_db(m, ref=np.max)


def mel_fig(y):
    S = mel_db(y)
    freqs = librosa.mel_frequencies(n_mels=96, fmax=SR // 2)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=SR, hop_length=256)
    fig = go.Figure(go.Heatmap(
        z=S, x=times, y=np.arange(96),
        colorscale=[[0, "#020a05"], [0.35, "#0D3B1E"], [0.7, "#00A843"], [1, "#00FF66"]],
        colorbar=dict(title="dB"), hovertemplate="t=%{x:.2f}s<br>mel bin %{y}<br>%{z:.1f} dB<extra></extra>"))
    ticks = np.linspace(0, 95, 7).astype(int)
    fig.update_yaxes(tickvals=ticks, ticktext=[f"{freqs[i]:.0f} Hz" for i in ticks], title="Frequency")
    fig.update_xaxes(title="Time (s)")
    return style_fig(fig, 320, "Mel-Spectrogram")


def mel_png(y):
    S = mel_db(y)
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor="#051C0D")
    ax.set_facecolor("#051C0D")
    ax.imshow(S, origin="lower", aspect="auto", cmap="Greens", extent=[0, len(y) / SR, 0, 8000])
    ax.set_xlabel("Time (s)", color="#B8F5CF")
    ax.set_ylabel("Hz (mel scale)", color="#B8F5CF")
    ax.tick_params(colors="#B8F5CF")
    for sp in ax.spines.values():
        sp.set_color("#00FF66")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


# ==============================================================================
# 5. MODEL + CONFIG LOADING
# ==============================================================================
def candidate_dirs(user_dir):
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    dirs = [user_dir, os.path.join(here, "models"), here, os.path.join(os.getcwd(), "models"), os.getcwd()]
    seen, out = set(), []
    for d in dirs:
        d = os.path.abspath(d) if d else None
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


@st.cache_resource(show_spinner="Loading models…")
def load_all_models(user_dir):
    """Dynamically load every *.joblib found. Returns (models, configs, errors, folder)."""
    models, configs, errors, folder = {}, {}, {}, None
    for d in candidate_dirs(user_dir):
        files = sorted(f for f in os.listdir(d) if f.endswith(".joblib"))
        if not files:
            continue
        folder = d
        for f in files:
            key = f[:-len(".joblib")]
            name = MODEL_DISPLAY.get(key, key.replace("_", " ").title())
            try:
                obj = joblib.load(os.path.join(d, f))
                if hasattr(obj, "pipeline") and hasattr(obj, "feature_cols"):  # ForestSoundModel
                    models[name] = dict(key=key, family="A", estimator=obj.pipeline, feature_cols=obj.feature_cols,
                                        labels=list(obj.class_names))
                else:
                    models[name] = dict(key=key, family="B", estimator=obj, feature_cols=None, labels=list(CLASSES_B_ORDER))
            except Exception as e:  # keep the app alive if one model fails
                errors[name] = f"{type(e).__name__}: {e}"
            cfg_path = os.path.join(d, f"{key}_config.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as fh:
                        configs[name] = json.load(fh)
                except Exception as e:
                    errors[f"{name} (config)"] = str(e)
        break
    ordered = {MODEL_DISPLAY[k]: models[MODEL_DISPLAY[k]] for k in MODEL_ORDER if MODEL_DISPLAY[k] in models}
    ordered.update({k: v for k, v in models.items() if k not in ordered})
    return ordered, configs, errors, folder


def predict_windows(name, spec, df_a, df_b):
    """Return (n_windows, 7) probabilities in canonical CLASSES order."""
    est = spec["estimator"]
    if spec["family"] == "A":
        X = df_a.reindex(columns=spec["feature_cols"]).fillna(0.0)
    else:
        X = df_b.values
    proba = est.predict_proba(X)
    out = np.zeros((proba.shape[0], len(CLASSES)))
    for col, cls_idx in enumerate(est.classes_):
        label = spec["labels"][int(cls_idx)]
        out[:, CLASSES.index(label)] = proba[:, col]
    return out


# ==============================================================================
# 6. DECISION LOGIC
# ==============================================================================
def summarize(probs_w, times, threats, threshold):
    """Clip-level verdict from per-window probabilities.

    Threat probability of a window = sum of the probabilities of the threat classes.
    The clip is flagged if ANY window reaches the threshold (a 1 s gunshot inside a 30 s
    clip must not be averaged away). Otherwise the label is the best non-threat class
    of the clip-average.
    """
    t_idx = [CLASSES.index(c) for c in threats]
    mean_p = probs_w.mean(0)
    threat_w = probs_w[:, t_idx].sum(1) if t_idx else np.zeros(len(probs_w))
    peak_i = int(threat_w.argmax())
    peak = float(threat_w[peak_i])
    is_threat = bool(t_idx) and peak >= threshold
    if is_threat:
        k = t_idx[int(np.argmax(probs_w[peak_i, t_idx]))]
        label, conf = CLASSES[k], peak
    else:
        safe_idx = [i for i in range(len(CLASSES)) if i not in t_idx] or list(range(len(CLASSES)))
        k = safe_idx[int(np.argmax(mean_p[safe_idx]))]
        label, conf = CLASSES[k], float(mean_p[k])
    return dict(label=label, conf=conf, is_threat=is_threat, peak=peak, peak_t=times[peak_i],
                mean_p=mean_p, threat_w=threat_w, mean_threat=float(threat_w.mean()))


def alert_panel(res, model_name):
    lab = res["label"]
    if res["is_threat"]:
        st.markdown(f"""
        <div class="alert alert-threat">
          <span class="badge">⚠ HIGH-RISK ACOUSTIC THREAT DETECTED</span>
          <div class="cls">{emoji(lab)} {pretty(lab).upper()}</div>
          <div class="pct">{res['peak'] * 100:.1f}%</div>
          <div class="sub">Threat probability (peak window @ {res['peak_t']:.1f} s) · {CLASS_META[lab][2]} · model: {model_name}</div>
        </div>""", unsafe_allow_html=True)
    else:
        extra = CLASS_META.get(lab, ("", "", ""))[2]
        st.markdown(f"""
        <div class="alert alert-safe">
          <span class="badge">✔ FOREST SAFE / NORMAL</span>
          <div class="cls">{emoji(lab)} {pretty(lab).upper()}</div>
          <div class="pct">{res['conf'] * 100:.1f}%</div>
          <div class="sub">Best-match class confidence · peak threat probability {res['peak'] * 100:.1f}% (below threshold) · {extra} · model: {model_name}</div>
        </div>""", unsafe_allow_html=True)


def class_prob_fig(mean_p, threats, title):
    order = np.argsort(mean_p)
    fig = go.Figure(go.Bar(
        x=mean_p[order] * 100, y=[f"{emoji(CLASSES[i])} {pretty(CLASSES[i])}" for i in order], orientation="h",
        marker=dict(color=[RED if CLASSES[i] in threats else NEON for i in order],
                    line=dict(color="rgba(255,255,255,0.25)", width=1)),
        text=[f"{mean_p[i] * 100:.1f}%" for i in order], textposition="outside", cliponaxis=False))
    fig.update_xaxes(title="Probability (%)", range=[0, 112])
    return style_fig(fig, 340, title)


def timeline_fig(probs_w, times, threats, threshold):
    t_idx = [CLASSES.index(c) for c in threats]
    labels = [f"{w:.0f}s" for w in times]
    threat_w = probs_w[:, t_idx].sum(1) if t_idx else np.zeros(len(times))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.08)
    fig.add_trace(go.Heatmap(z=probs_w.T * 100, x=labels, y=[pretty(c) for c in CLASSES],
                             colorscale=[[0, "#041509"], [0.5, "#0b7a38"], [1, "#00FF66"]], zmin=0, zmax=100,
                             colorbar=dict(title="%", len=0.55, y=0.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=labels, y=threat_w * 100, mode="lines+markers", name="Threat prob.",
                             line=dict(color=RED, width=2), marker=dict(size=8, color=RED)), row=2, col=1)
    fig.add_hline(y=threshold * 100, line=dict(color=AMBER, dash="dash"), row=2, col=1)
    fig.update_yaxes(title_text="Threat %", range=[0, 105], row=2, col=1)
    fig.update_xaxes(title_text="Window start", row=2, col=1)
    return style_fig(fig, 430, "Per-window timeline")


# ==============================================================================
# 7. SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("## 🌲 EcoGuard")
    st.caption("Acoustic guardians for the forest.")
    st.markdown("---")
    st.markdown("### 🎚️ Input")
    source = st.radio("Audio source", ["Upload file", "Record live", "Sample / test signal"], key="source")
    max_windows = st.slider("Max 3-second windows to analyse", 1, 40, 20, key="max_windows",
                            help="Long recordings are split into consecutive 3 s windows (the length used in training).")

    st.markdown("### 🧠 Model")
    model_dir = st.text_input("Model folder", value="models", key="model_dir",
                              help="Folder with *.joblib and *_config.json. Falls back to the app folder.")
    MODELS, CONFIGS, LOAD_ERRORS, MODEL_FOLDER = load_all_models(model_dir)
    if MODELS:
        selected_model = st.selectbox("Model for single prediction", list(MODELS.keys()), key="selected_model")
    else:
        selected_model = None
        st.error("No .joblib models found.")

    st.markdown("### 🚨 Alert policy")
    threat_classes = st.multiselect("Classes treated as threats", CLASSES, default=DEFAULT_THREATS,
                                    format_func=pretty, key="threat_classes")
    threshold = st.slider("Threat alert threshold", 0.05, 0.99, 0.50, 0.01, key="threshold",
                          help="Alert if the summed threat-class probability of any window reaches this value.")

    st.markdown("---")
    st.markdown(f"**Models loaded:** {len(MODELS)}/6")
    for n in MODELS:
        st.markdown(f'<span class="pill">{n}</span>', unsafe_allow_html=True)
    for n, e in LOAD_ERRORS.items():
        st.warning(f"{n}: {e}")
    if MODEL_FOLDER:
        st.caption(f"Folder: {MODEL_FOLDER}")

# ==============================================================================
# 8. HEADER
# ==============================================================================
st.markdown(f"""
<div class="eg-hero">
  <h1>🌲 EcoGuard <span style="font-size:.6em;opacity:.8">| Real-Time Acoustic Threat Detection System</span></h1>
  <p>Detect chainsaws, gunshots and fire in forest soundscapes with six tuned ML models, side-by-side.</p>
</div>""", unsafe_allow_html=True)

# ==============================================================================
# 9. AUDIO INPUT
# ==============================================================================
audio_bytes, audio_name = None, None
if source == "Upload file":
    up = st.file_uploader("Drop a forest recording (WAV · MP3 · OGG)", type=["wav", "mp3", "ogg"], key="uploader")
    if up is not None:
        audio_bytes, audio_name = up.getvalue(), up.name
elif source == "Record live":
    if hasattr(st, "audio_input"):
        rec = st.audio_input("Record from your microphone", key="recorder")
        if rec is not None:
            audio_bytes, audio_name = rec.getvalue(), "recording.wav"
    else:
        st.info("Live recording needs Streamlit ≥ 1.39 (`pip install -U streamlit`). Use file upload instead.")
else:
    options = {}
    for d in candidate_dirs("samples"):
        sd = os.path.join(d, "samples") if os.path.basename(d) != "samples" else d
        if os.path.isdir(sd):
            for f in sorted(os.listdir(sd)):
                if f.lower().endswith((".wav", ".mp3", ".ogg")):
                    options[f"File · {f}"] = os.path.join(sd, f)
    synth = synth_samples()
    pick = st.selectbox("Choose a sample", list(options.keys()) + list(synth.keys()), key="sample_pick")
    if pick in options:
        with open(options[pick], "rb") as fh:
            audio_bytes, audio_name = fh.read(), os.path.basename(options[pick])
    else:
        audio_bytes, audio_name = _to_wav_bytes(synth[pick]), "synthetic.wav"
        st.caption("⚠️ Synthetic signals only exercise the pipeline — they are not real recordings, so the "
                   "model outputs on them are not meaningful. Put real clips in a `samples/` folder to list them here.")

if audio_bytes is None:
    st.markdown("""<div class="eg-card"><h3>Awaiting audio…</h3>
    Upload a recording, record from the microphone, or pick a sample in the sidebar to start the analysis.</div>""",
                unsafe_allow_html=True)
    st.stop()

if not MODELS:
    st.error("No trained models were found. Place the *.joblib files in a `models/` folder next to this script.")
    st.stop()

try:
    y = load_audio(audio_bytes, audio_name)
except Exception as e:
    st.error(f"Could not decode this audio file ({type(e).__name__}: {e}). Try WAV, or install ffmpeg for MP3/OGG.")
    st.stop()
if len(y) < SR // 4:
    st.error("The audio is too short (< 0.25 s).")
    st.stop()

audio_key = hashlib.md5(y.tobytes()).hexdigest() + f"_{max_windows}"
with st.spinner("Extracting acoustic features (MFCC · spectral · ZCR · RMS · flux)…"):
    w_times, w_wins, FA, FB = compute_features(y, max_windows)

st.session_state.setdefault("probs", {})
st.session_state["probs"].setdefault(audio_key, {})
CACHE = st.session_state["probs"][audio_key]


def get_probs(name):
    if name not in CACHE:
        CACHE[name] = predict_windows(name, MODELS[name], FA, FB)
    return CACHE[name]


c1, c2, c3, c4 = st.columns(4)
kpi(c1, f"{len(y) / SR:.1f}s", "Clip length")
kpi(c2, f"{len(w_wins)}", "Windows analysed")
kpi(c3, f"{len(MODELS)}", "Models ready")
kpi(c4, f"{FA.shape[1]}", "Features / window")
if len(y) / SR > len(w_wins) * 3 + 1:
    st.caption(f"Only the first {len(w_wins) * 3} s were analysed (raise the window limit in the sidebar to cover more).")

tab_main, tab_cmp, tab_hp, tab_dl = st.tabs(
    ["🎧 Analysis & Prediction", "⚖️ Predict All & Compare", "🧬 Hyperparameters", "🧠 Deep Learning"])

# ==============================================================================
# TAB 1 — ANALYSIS + SINGLE MODEL
# ==============================================================================
with tab_main:
    st.audio(audio_bytes)
    show_plot(waveform_fig(y, w_times), key="wave")
    show_plot(mel_fig(y), key="mel")
    st.download_button("⬇ Download spectrogram (PNG)", mel_png(y), "ecoguard_mel_spectrogram.png", "image/png")

    st.markdown(f"### 🎯 Prediction · {selected_model}")
    probs_sel = get_probs(selected_model)
    res = summarize(probs_sel, w_times, threat_classes, threshold)
    alert_panel(res, selected_model)
    a, b = st.columns([1, 1])
    with a:
        show_plot(class_prob_fig(res["mean_p"], threat_classes, "Clip-average class probabilities"), key="clsbar")
    with b:
        show_plot(timeline_fig(probs_sel, w_times, threat_classes, threshold), key="timeline")
    with st.expander("🔬 Extracted feature vector (first window)"):
        fa = FA.iloc[0]
        show_df(pd.DataFrame({"feature": fa.index, "value": fa.values}).head(60))
        st.caption(f"{FA.shape[1]} aggregated features (mean/std/min/max of 73 frame-level descriptors).")

# ==============================================================================
# TAB 2 — PREDICT ALL & COMPARE
# ==============================================================================
with tab_cmp:
    st.markdown("### ⚖️ Run every loaded model on this clip")
    if st.button("🚀 Predict All & Compare Models", key="run_all"):
        prog = st.progress(0.0, text="Running inference…")
        for i, n in enumerate(MODELS):
            get_probs(n)
            prog.progress((i + 1) / len(MODELS), text=f"{n} done")
        prog.empty()
        st.session_state["cmp_done"] = audio_key

    if st.session_state.get("cmp_done") != audio_key:
        st.info("Press the button to compare all models on the current clip.")
    else:
        rows, summ = [], {}
        for n in MODELS:
            r = summarize(get_probs(n), w_times, threat_classes, threshold)
            summ[n] = r
            cfg = CONFIGS.get(n, {})
            rows.append({
                "Model": n,
                "Prediction": f"{emoji(r['label'])} {pretty(r['label'])}",
                "Confidence (%)": round(r["conf"] * 100, 2),
                "Peak threat prob. (%)": round(r["peak"] * 100, 2),
                "Peak @ (s)": round(r["peak_t"], 1),
                "Status": "🚨 THREAT" if r["is_threat"] else "✅ SAFE",
                "Val macro-F1": cfg.get("val_macro_f1", np.nan),
            })
        # F1-weighted soft-voting ensemble
        wts = np.array([CONFIGS.get(n, {}).get("val_macro_f1", 1.0) for n in MODELS], dtype=float)
        ens = sum(w * get_probs(n) for w, n in zip(wts, MODELS)) / wts.sum()
        er = summarize(ens, w_times, threat_classes, threshold)
        rows.append({"Model": "★ Ensemble (F1-weighted)", "Prediction": f"{emoji(er['label'])} {pretty(er['label'])}",
                     "Confidence (%)": round(er["conf"] * 100, 2), "Peak threat prob. (%)": round(er["peak"] * 100, 2),
                     "Peak @ (s)": round(er["peak_t"], 1), "Status": "🚨 THREAT" if er["is_threat"] else "✅ SAFE",
                     "Val macro-F1": np.nan})
        df_cmp = pd.DataFrame(rows)

        votes = sum(r["is_threat"] for r in summ.values())
        st.markdown(f"**Consensus:** {votes} of {len(summ)} models flag a threat · ensemble verdict below.")
        alert_panel(er, "Ensemble (F1-weighted)")

        st.markdown("#### 📋 Unified comparison table")
        show_df(df_cmp, column_config={
            "Confidence (%)": st.column_config.ProgressColumn("Confidence (%)", min_value=0, max_value=100, format="%.1f"),
            "Peak threat prob. (%)": st.column_config.ProgressColumn("Peak threat prob. (%)", min_value=0, max_value=100, format="%.1f"),
            "Val macro-F1": st.column_config.NumberColumn("Val macro-F1", format="%.4f"),
        })
        st.download_button("⬇ Download comparison (CSV)", df_cmp.to_csv(index=False).encode("utf-8"),
                           "ecoguard_model_comparison.csv", "text/csv")

        st.markdown("#### 📊 Confidence across models")
        d = df_cmp.iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=d["Model"], x=d["Confidence (%)"], orientation="h", name="Confidence in prediction",
                             marker=dict(color=[NEON if "SAFE" in s else ORANGE for s in d["Status"]],
                                         line=dict(color="rgba(255,255,255,.3)", width=1)),
                             text=[f"{p} · {c:.1f}%" for p, c in zip(d["Prediction"], d["Confidence (%)"])],
                             textposition="inside", insidetextanchor="start"))
        fig.add_trace(go.Bar(y=d["Model"], x=d["Peak threat prob. (%)"], orientation="h", name="Peak threat probability",
                             marker=dict(color=RED), opacity=0.75,
                             text=[f"{v:.1f}%" for v in d["Peak threat prob. (%)"]], textposition="outside"))
        fig.update_layout(barmode="group")
        fig.update_xaxes(title="Probability (%)", range=[0, 115])
        show_plot(style_fig(fig, 430, "Model confidence vs. threat probability"), key="cmp_bar")

        st.markdown("#### 🔥 Class probabilities by model (clip average)")
        names = list(MODELS) + ["★ Ensemble"]
        mat = np.vstack([summ[n]["mean_p"] for n in MODELS] + [er["mean_p"]]) * 100
        hm = go.Figure(go.Heatmap(z=mat, x=[pretty(c) for c in CLASSES], y=names,
                                  colorscale=[[0, "#041509"], [0.5, "#0b7a38"], [1, "#00FF66"]], zmin=0, zmax=100,
                                  text=np.round(mat, 1), texttemplate="%{text}", colorbar=dict(title="%")))
        show_plot(style_fig(hm, 380), key="cmp_heat")
        if len({s["label"] for s in summ.values()}) > 1:
            st.warning("Models disagree on the class — review the heatmap before trusting a single model.")

# ==============================================================================
# TAB 3 — HYPERPARAMETERS
# ==============================================================================
META_KEYS = {"model_name", "owner", "val_macro_f1", "cv_macro_f1", "cv", "val_fold", "train_folds", "class_names",
             "n_features", "sample_rate", "window_samples", "seed", "versions", "created", "api", "requires",
             "catboost_version", "xgboost_version", "sklearn_version", "best_params"}


def get_params(cfg):
    """Tuned params from a config: `best_params` if present, else flat top-level estimator keys (SVM)."""
    raw = cfg.get("best_params") or {k: v for k, v in cfg.items() if k not in META_KEYS}
    return {k.replace("clf__", ""): v for k, v in raw.items()}


with tab_hp:
    st.markdown("### 🧬 Tuned hyperparameters (from `*_config.json`)")
    if not CONFIGS:
        st.warning("No *_config.json files found next to the models.")
    else:
        sel = st.selectbox("Inspect model", list(CONFIGS.keys()),
                           index=list(CONFIGS.keys()).index(selected_model) if selected_model in CONFIGS else 0,
                           key="hp_model")
        cfg = CONFIGS[sel]
        m1, m2, m3, m4 = st.columns(4)
        kpi(m1, f"{cfg.get('val_macro_f1', float('nan')):.4f}", "Val macro-F1")
        kpi(m2, f"{cfg['cv_macro_f1']:.4f}" if "cv_macro_f1" in cfg else "—", "CV macro-F1")
        kpi(m3, str(cfg.get("owner", "—")).split("—")[0].strip(), "Owner")
        kpi(m4, str(cfg.get("created", "—"))[:10], "Trained")
        pa, pb = st.columns([1, 1])
        with pa:
            params = get_params(cfg)
            st.markdown("**Best parameters**")
            show_df(pd.DataFrame({"Parameter": list(params.keys()),
                                  "Value": [json.dumps(v) if not isinstance(v, str) else v for v in params.values()]}))
        with pb:
            st.markdown("**Training setup**")
            setup = {k: (json.dumps(v) if isinstance(v, (dict, list)) else str(v)) for k, v in cfg.items()
                     if k in ("cv", "train_folds", "val_fold", "n_features", "sample_rate", "window_samples",
                              "class_weight", "seed", "api", "versions", "sklearn_version", "xgboost_version",
                              "catboost_version")}
            show_df(pd.DataFrame({"Setting": list(setup.keys()), "Value": list(setup.values())}))
        with st.expander("Raw JSON"):
            st.json(cfg)

        st.markdown("#### 🔀 All models side-by-side")
        allp = {n: get_params(c) for n, c in CONFIGS.items()}
        keys = sorted({k for p in allp.values() for k in p})
        grid = pd.DataFrame({"Parameter": keys})
        for n, p in allp.items():
            grid[n] = [("" if k not in p else (f"{p[k]:.4g}" if isinstance(p[k], float) else str(p[k]))) for k in keys]
        show_df(grid)

        st.markdown("#### 📈 Tuned quality vs. prediction on the current clip")
        perf = []
        for n, c in CONFIGS.items():
            row = {"Model": n, "Val macro-F1": c.get("val_macro_f1", np.nan)}
            if n in MODELS and n in CACHE:
                r = summarize(CACHE[n], w_times, threat_classes, threshold)
                row.update({"Prediction": pretty(r["label"]), "Confidence (%)": round(r["conf"] * 100, 1),
                            "Peak threat (%)": round(r["peak"] * 100, 1)})
            perf.append(row)
        pdf = pd.DataFrame(perf).sort_values("Val macro-F1", ascending=False)
        show_df(pdf)
        figf = go.Figure(go.Bar(x=pdf["Model"], y=pdf["Val macro-F1"], marker=dict(color=NEON),
                                text=[f"{v:.3f}" for v in pdf["Val macro-F1"]], textposition="outside"))
        figf.update_yaxes(range=[max(0, pdf["Val macro-F1"].min() - 0.1), 1.0], title="Validation macro-F1 (fold 1)")
        show_plot(style_fig(figf, 320, "Validation macro-F1 per model"), key="hp_f1")
        if "Confidence (%)" not in pdf.columns:
            st.caption("Run 'Predict All & Compare Models' to add this clip's predictions to the table.")
        st.info("ℹ️ CatBoost / SVM / XGBoost configs list `class_names` in the order of the project's CLASS_LIST, but "
                "those models were trained on sklearn LabelEncoder (alphabetical) labels. EcoGuard uses the "
                "alphabetical order, matching the notebook's training code.")

# ==============================================================================
# TAB 4 — DEEP LEARNING (extensible)
# ==============================================================================
DL_KERAS_EXAMPLE = '''import tensorflow as tf
from tensorflow.keras import layers, models

# 1-D CNN on the raw 3 s waveform (48000 samples @ 16 kHz)
def build_1d_cnn(n_classes=7, win=48000):
    inp = layers.Input(shape=(win, 1))
    x = layers.Conv1D(16, 80, strides=4, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(4)(x)
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(4)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(4)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inp, out)

model = build_1d_cnn()
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# ... model.fit(...)
model.save("models/deep_learning.keras")   # EcoGuard auto-detects this file'''

DL_ARCH = pd.DataFrame({
    "Layer": ["Input", "Conv1D(16, k=80, s=4) + BN + MaxPool(4)", "Conv1D(32, k=3) + BN + MaxPool(4)",
              "Conv1D(64, k=3) + BN + MaxPool(4)", "GlobalAveragePooling1D", "Dropout(0.3)", "Dense(7, softmax)"],
    "Output shape": ["(48000, 1)", "(3000, 16)", "(187, 32)", "(11, 64)", "(64)", "(64)", "(7)"],
})


@st.cache_resource(show_spinner="Loading deep-learning model…")
def load_dl_model(path, mtime):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".keras", ".h5"):
        try:
            from tensorflow import keras
        except ImportError:
            import keras  # standalone Keras 3
        return "keras", keras.models.load_model(path, compile=False)
    if ext in (".pt", ".pth"):
        import torch
        return "torch", torch.jit.load(path, map_location="cpu").eval()
    raise ValueError(f"Unsupported extension {ext}")


def _fit_axis(a, n, axis):
    if a.shape[axis] >= n:
        return np.take(a, np.arange(n), axis=axis)
    pad = [(0, 0)] * a.ndim
    pad[axis] = (0, n - a.shape[axis])
    return np.pad(a, pad)


def dl_predict(kind, model, wins, mode, layout, n_mels, frames):
    if mode == "Raw waveform":
        X = np.stack(wins).astype("float32")
        if kind == "keras":
            shp = model.input_shape[1:]
            X = _fit_axis(X, int(shp[0]), 1)
            if len(shp) == 2:
                X = X[..., None]
        else:
            X = X[:, None, :]
    else:
        mels = np.stack([librosa.power_to_db(librosa.feature.melspectrogram(
            y=w, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=n_mels)) for w in wins]).astype("float32")  # (n, mels, T)
        if kind == "keras":
            shp = model.input_shape[1:]
            if layout.startswith("(frames"):
                mels = mels.transpose(0, 2, 1)
            mels = _fit_axis(_fit_axis(mels, int(shp[0]), 1), int(shp[1]), 2)
            X = mels[..., None] if len(shp) == 3 else mels
        else:
            X = _fit_axis(mels, frames, 2)[:, None, :, :]
    if kind == "keras":
        p = np.asarray(model.predict(X, verbose=0))
    else:
        import torch
        with torch.no_grad():
            p = model(torch.from_numpy(X)).numpy()
    if not np.allclose(p.sum(1), 1.0, atol=1e-3):  # logits -> softmax
        e = np.exp(p - p.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
    return p


with tab_dl:
    st.markdown("### 🧠 Deep Learning models (extensible)")
    st.markdown('<div class="eg-card">No deep-learning model ships with this project yet. Drop a '
                '<code>.keras</code>, <code>.h5</code> or TorchScript <code>.pt</code> file below (or save it as '
                '<code>models/deep_learning.keras</code>) and EcoGuard will run it on the same 3 s windows.</div>',
                unsafe_allow_html=True)
    with st.expander("🏗️ Reference 1-D CNN architecture (placeholder) & export recipe"):
        show_df(DL_ARCH)
        st.code(DL_KERAS_EXAMPLE, language="python")
        st.caption("Any architecture works as long as it outputs one probability (or logit) per class.")

    dl_path = None
    for d in candidate_dirs(model_dir):
        for ext in (".keras", ".h5", ".pt", ".pth"):
            p = os.path.join(d, f"deep_learning{ext}")
            if os.path.exists(p):
                dl_path = p
                break
        if dl_path:
            break
    dl_up = st.file_uploader("Upload a trained DL model", type=["keras", "h5", "pt", "pth"], key="dl_up")
    if dl_up is not None:
        tmp = os.path.join(tempfile.gettempdir(), f"ecoguard_{hashlib.md5(dl_up.getvalue()).hexdigest()[:8]}_{dl_up.name}")
        if not os.path.exists(tmp):
            with open(tmp, "wb") as fh:
                fh.write(dl_up.getvalue())
        dl_path = tmp

    if not dl_path:
        st.info("No deep-learning model detected — the six classical models above are fully functional.")
    else:
        st.success(f"Model file: {os.path.basename(dl_path)}")
        o1, o2 = st.columns(2)
        mode = o1.selectbox("Expected input", ["Raw waveform", "Log-Mel spectrogram"], key="dl_mode")
        class_txt = o2.text_input("Class order (comma-separated, output-index order)", ", ".join(CLASSES_B_ORDER), key="dl_cls")
        layout, n_mels, frames = "(mels, frames)", 64, 301
        if mode == "Log-Mel spectrogram":
            q1, q2, q3 = st.columns(3)
            layout = q1.selectbox("Keras layout", ["(mels, frames[, 1])", "(frames, mels[, 1])"], key="dl_layout")
            n_mels = q2.number_input("n_mels", 16, 256, 64, key="dl_nmels")
            frames = q3.number_input("frames (TorchScript)", 50, 600, 301, key="dl_frames")
        if st.button("Run deep-learning inference", key="dl_run"):
            try:
                labels = [c.strip() for c in class_txt.split(",") if c.strip()]
                if sorted(labels) != sorted(CLASSES):
                    raise ValueError(f"Class list must contain exactly: {', '.join(CLASSES)}")
                kind, dlm = load_dl_model(dl_path, os.path.getmtime(dl_path))
                raw = dl_predict(kind, dlm, w_wins, mode, layout, int(n_mels), int(frames))
                if raw.shape[1] != len(labels):
                    raise ValueError(f"Model outputs {raw.shape[1]} classes but {len(labels)} names were given.")
                P = np.zeros((raw.shape[0], len(CLASSES)))
                for j, lab in enumerate(labels):
                    P[:, CLASSES.index(lab)] = raw[:, j]
                r = summarize(P, w_times, threat_classes, threshold)
                alert_panel(r, "Deep Learning")
                show_plot(class_prob_fig(r["mean_p"], threat_classes, "Deep-learning class probabilities"), key="dl_bar")
            except ImportError as e:
                st.error(f"Missing dependency: {e}. Install tensorflow/keras or torch to use this model type.")
            except Exception as e:
                st.error(f"Could not run the model: {type(e).__name__}: {e}")
        st.caption("The input adapter is generic (waveform or log-mel). If your network expects different "
                   "preprocessing, adjust `dl_predict()` in this script.")

st.markdown("---")
st.caption("EcoGuard · Forest Threat Sound Detection · classical models trained on 3 s windows @ 16 kHz")
