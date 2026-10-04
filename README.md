# 🌲 EcoGuard — Forest Threat Sound Detection

> **Real-time acoustic monitoring for forests: detect chainsaws, gunshots, and fire using classical ML and deep learning — side by side.**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Team & Roles](#2-team--roles)
3. [Datasets](#3-datasets)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Feature Extraction](#5-feature-extraction)
6. [Models](#6-models)
7. [Training Methodology](#7-training-methodology)
8. [Evaluation & Results](#8-evaluation--results)
9. [Streamlit App (EcoGuard)](#9-streamlit-app-ecoguard)
10. [Project Structure](#10-project-structure)
11. [Installation & Quick Start](#11-installation--quick-start)
12. [Usage Guide](#12-usage-guide)
13. [Reproducibility](#13-reproducibility)
14. [Dependencies](#14-dependencies)

---

## 1. Project Overview

EcoGuard is an end-to-end acoustic threat-detection system designed to protect forests in real time. A microphone placed in the field streams short audio windows to the pipeline; the pipeline classifies each window into one of **7 classes** and raises an alert if a threat is detected.

**Threat classes:** chainsaw/engine, fire, gunshot  
**Benign classes:** human voice, natural forest sounds, rain/thunder, unknown background

The project integrates **six tuned classical ML models** and a **CRNN deep learning model** into a single Streamlit dashboard, allowing side-by-side comparison and F1-weighted ensemble voting.

---

## 2. Team & Roles

| Role | Engineer | Deliverables |
|---|---|---|
| **Data & Signals Engineer** | Eng. Israa Hamdy | Dataset discovery, cleaning, feature extraction, shared evaluation harness |
| **Developer A** — Baseline & Tree Ensembles | Eng. Sajy | Logistic Regression, Random Forest, LightGBM, SHAP analysis, ablation study |
| **Developer B** — Advanced Boosting & SVM | Eng. Jomanah | SVM-RBF, XGBoost, CatBoost, error analysis, latency profiling |
| **Deep Learning Lead** | Eng. Mohamed Mansour | CRNN (CNN + BiGRU), spectrogram input pipeline, final evaluation |

---

## 3. Datasets

Eight Kaggle datasets were merged and mapped to the 7 project classes.

### 3.1 Source Datasets

| # | Kaggle Dataset | Key Sounds |
|---|---|---|
| 1 | `ivanj0/audiodata` | General audio classifier (gunshot, hi-hat, shatter…) |
| 2 | `karolpiczak/ESC-50` | Environmental sounds (50 classes) |
| 3 | `irmiot22/fsc22-dataset` | Forest soundscapes |
| 4 | `emrahaydemr/gunshot-audio-dataset` | Rifle, AK-47, M16, pistol shots |
| 5 | `rupakroy/urban-sound-8k` | Urban audio (siren, drilling, street music…) |
| 6 | `forestprotection/forest-wild-fire-sound-dataset` | Wildfire crackling |
| 7 | `rfcx-species-audio-16000` | Rainforest species (birds, insects, frogs) |
| 8 | Fire and Forest Audio Classification | Combined fire & forest categories |

**Total raw files discovered:** 26,702 audio clips across all sources.

### 3.2 Class Mapping

| Target Class | Source Sounds |
|---|---|
| `natural` | Birds, wind, insects, crickets, frogs, sea waves, RFCx species |
| `rain_thunder` | Rain, thunderstorm, heavy rain |
| `gunshot` | Rifle, AK-47, M16, pistol, gunfire sounds |
| `fire` | Wildfire crackling, forest fire, camp fire |
| `chainsaw_engine` | Chainsaw, generator, engine idling, vehicle engine |
| `human_voice` | Speech, shouting, laughter, crying, coughing |
| `unknown` | Siren, drilling, jackhammer, car horn, fireworks, helicopter, street music, axe |

### 3.3 Cleaning & Balancing

- All audio is resampled to **mono, 16 kHz, padded/trimmed to 3 seconds (48,000 samples)**.
- Majority classes (e.g., `natural` at 7,139 clips, `unknown` at 6,579 clips) are **capped at 1,500 clips per (class, source-dataset)** pair to prevent domination.
- Minority safety-critical classes (`fire` — 140 clips; `rain_thunder` — 80 clips) are **kept in full** without capping.
- Invalid files (corrupt, too short, unreadable) are discarded.

### 3.4 Train / Validation / Test Split

Splitting is done with **StratifiedGroupKFold (n_splits=10, seed=42)** grouped by source recording (`group_id = source_dataset::orig_filename`) to prevent clip leakage from the same recording appearing in both train and test.

| Partition | Folds | Purpose |
|---|---|---|
| **Test** | Fold 0 | Frozen; SHA-256 tripwire enforced; never used for selection |
| **Validation** | Fold 1 | Model selection and hyperparameter choice |
| **Training** | Folds 2–9 | Model fitting (~80% of corpus) |

---

## 4. Pipeline Architecture

```
Raw Audio (8 Kaggle datasets)
          │
          ▼
 ┌─────────────────────┐
 │  Data & Signals Eng │  Resample · clean · stratified split
 └─────────────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  Feature Extraction │  Frame-level → aggregated (classical)
 │                     │  Log-mel spectrogram (deep learning)
 └─────────────────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌───────┐  ┌─────────┐
│ Dev A │  │  Dev B  │     Deep Learning Lead
│ LR/RF │  │SVM/XGB/ │  ┌─────────────────────┐
│ LGBM  │  │CatBoost │  │  CRNN (CNN+BiGRU)   │
└───────┘  └─────────┘  └─────────────────────┘
    │            │                │
    └────────────┴────────────────┘
                 │
                 ▼
      ┌──────────────────┐
      │  Shared Eval     │  Macro-F1 · ROC · PR · Noise robustness
      └──────────────────┘
                 │
                 ▼
      ┌──────────────────┐
      │  EcoGuard App    │  Streamlit dashboard · ensemble voting
      └──────────────────┘
```

---

## 5. Feature Extraction

### 5.1 Classical ML Features (Developer A & B)

Frame-level analysis settings: **N_FFT = 512, HOP = 160** (32 ms frames, 10 ms hop at 16 kHz).

| Feature Group | Features | Dim/frame |
|---|---|---|
| MFCC | Coefficients 0–19 | 20 |
| Delta MFCC | 1st-order derivative | 20 |
| Delta-Delta MFCC | 2nd-order derivative | 20 |
| Spectral Centroid | Centre of mass | 1 |
| Spectral Bandwidth | Spread around centroid | 1 |
| Spectral Rolloff | 85% energy cutoff | 1 |
| Spectral Contrast | Sub-band peak-valley ratio | 7 |
| Zero-Crossing Rate | Sign-change rate | 1 |
| RMS Energy | Root-mean-square amplitude | 1 |
| Spectral Flux | Frame-to-frame magnitude change | 1 |

Each frame-level feature is aggregated over the 3-second window with **mean, std, min, max** → **~292 total features per clip**, saved to `features.parquet`.

> **Two extractors exist** (due to the split development):
> - **Dev A** (`forest_model_api.ForestSoundModel`): uses `frame_length=N_FFT` for ZCR, prepends a leading `0` to spectral flux, 0-based MFCC naming.
> - **Dev B**: uses default ZCR frame length, no flux padding, 1-based feature naming.
>
> The Streamlit app reproduces both extractors exactly so each model is always fed the features it was trained on.

### 5.2 Deep Learning Features (CRNN)

The `ForestDataset` PyTorch class provides three parallel modalities:

| Modality | Shape | Use |
|---|---|---|
| `waveform` | `(48000,)` | Raw 1D audio |
| `logmel` | `(64, T)` | 64-mel log-mel spectrogram (N_FFT=1024, HOP=320) |
| `logmel_img` | `(3, 224, 224)` | ImageNet-normalized 3-channel spectrogram for transfer learning |

**Training augmentations** (train split only):
- Additive Gaussian noise at random SNR (5–30 dB), 30% probability
- Random gain (−6 to +6 dB)
- Circular time shift up to ±30% of clip length

---

## 6. Models

### 6.1 Developer A — Baseline & Tree Ensembles (Eng. Sajy)

| Model | Tuning | Notes |
|---|---|---|
| **Logistic Regression** | — (mandatory baseline) | L2 regularisation, `class_weight='balanced'` |
| **Random Forest** | 8-trial `RandomizedSearchCV` | n_estimators, max_depth, min_samples_split/leaf, max_features |
| **LightGBM** | 15-trial `RandomizedSearchCV` | n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha/lambda |

**Additional analysis:** SHAP per-class feature contributions, feature-group importance (Gini), and a 9-configuration ablation study quantifying the contribution of each feature family.

### 6.2 Developer B — Advanced Boosting & SVM (Eng. Jomanah)

| Model | Tuning | Notes |
|---|---|---|
| **SVM (RBF)** | Full grid: C ∈ {1,10,100} × γ ∈ {scale, 0.001, 0.01} | 9 configs |
| **XGBoost** | 30-trial random search | n_estimators up to 1500, `early_stopping_rounds=40` |
| **CatBoost** | Full grid: depth × learning_rate × l2_leaf_reg | 27 configs, `early_stopping_rounds=50` |

**Additional analysis:** per-class error analysis, latency benchmarking (mean, P50, P95, P99, max in ms).

### 6.3 Deep Learning — CRNN (Eng. Mohamed Mansour)

```
Input: (batch, 1, 64, T)   ← single-channel log-mel spectrogram

CNN Frontend
  Conv2d(1→32, 3×3, pad=1) → BatchNorm2d(32) → ReLU
  Conv2d(32→64, 3×3, pad=1) → BatchNorm2d(64) → ReLU
  MaxPool2d(kernel=(2,1))           # halves mel axis, keeps time axis
  Conv2d(64→128, 3×3, pad=1) → BatchNorm2d(128) → ReLU
  MaxPool2d(kernel=(2,1))           # mel: 64 → 16

Reshape
  (batch, 128, 16, T) → permute → (batch, T, 2048)

Bidirectional GRU
  input_size=2048, hidden_size=256, num_layers=2
  batch_first=True, dropout=0.3, bidirectional=True
  output: (batch, T, 512)

Temporal pooling:  mean over T → (batch, 512)

Classifier:  Linear(512 → 7)
```

**Training configuration:**

| Setting | Value |
|---|---|
| Optimizer | Adam, lr = 5 × 10⁻⁴ |
| Loss | Weighted CrossEntropyLoss (class weights from `compute_class_weight`) |
| LR schedule | `ReduceLROnPlateau` (mode=max, patience=5, factor=0.5) |
| Early stopping | patience = 5 epochs on val macro-F1 |
| Gradient clipping | max_norm = 1.0 |
| Batch size | 32 |
| Max epochs | 9 |
| Checkpoint | Saved on every val macro-F1 improvement |

---

## 7. Training Methodology

### Anti-leakage rules (enforced throughout)
- `StandardScaler` lives inside every sklearn `Pipeline` — fitted only on training folds.
- All inner CV for hyperparameter tuning uses `GroupKFold(n_splits=3)` on `group_id` — standard KFold is prohibited.
- Model selection is based **exclusively** on Fold 1 (validation) macro-F1.
- The frozen test set (Fold 0) is accessed **once per developer**, only in the "Final Evaluation" cell.
- A SHA-256 tripwire on `test_manifest.csv` prevents accidental test set modification.

### Class imbalance handling
- Developer A & B: `class_weight='balanced'` in all classifiers; XGBoost uses `compute_sample_weight('balanced')`.
- Deep learning: `compute_class_weight('balanced')` used to weight `CrossEntropyLoss`.

### Experiment logging
All hyperparameter search runs are logged to `experiments_log.csv` with CV macro-F1 and validation macro-F1, enabling full reproducibility.

---

## 8. Evaluation & Results

### Shared evaluation harness (Eng. Israa Hamdy)

A model-agnostic harness evaluates **every** model — classical and deep — using the same functions for direct comparability.

**API contract:** every model exposes  
`predict(window: np.ndarray[48000]) → dict[class_name: str, probability: float]`

**Evaluation outputs per model:**
- Confusion matrix on frozen test set (Fold 0)
- Per-class classification report (precision, recall, F1, support)
- One-vs-rest ROC curves with AUC per class
- One-vs-rest Precision-Recall curves with Average Precision per class
- **Noise robustness test:** accuracy at SNR = 5, 10, 20 dB on 200 test clips
- **Latency profile:** mean, P50, P95, P99, max inference time (ms) per 3-second window

### Primary metric
**Macro-F1** — all 7 classes weighted equally, favouring models that generalise across rare safety-critical classes (fire, rain_thunder).

---

## 9. Streamlit App (EcoGuard)

The `app.py` file is a single-file Streamlit dashboard that loads all trained models and provides a real-time classification interface.

### Tabs

| Tab | What it shows |
|---|---|
| 🎧 **Analysis & Prediction** | Waveform, mel-spectrogram, single-model prediction with alert panel |
| ⚖️ **Predict All & Compare** | All 6 + CRNN models run in one click; F1-weighted ensemble; comparison table & heatmap |
| 🧬 **Hyperparameters** | Tuned params and training setup from `*_config.json`; per-model validation F1 bar chart |
| 🧠 **Deep Learning** | CRNN-specific results, architecture KPIs, per-window timeline |

### Key features
- **3-second window analysis** — long recordings are automatically split into consecutive windows.
- **Configurable alert policy** — choose which classes count as threats and set the probability threshold.
- **Live microphone recording** via `st.audio_input` (Streamlit ≥ 1.39).
- **F1-weighted soft-voting ensemble** — models with higher validation macro-F1 get proportionally more weight.
- **Near-miss warning** — flags when peak threat probability is 60% of the threshold but below it.
- **Downloadable outputs** — mel-spectrogram PNG, per-model comparison CSV.

### Model file conventions

```
models/
├── catboost.joblib
├── catboost_config.json
├── lightgbm.joblib
├── lightgbm_config.json
├── logistic_regression.joblib
├── logistic_regression_config.json
├── random_forest.joblib
├── random_forest_config.json
├── svm_rbf.joblib
├── svm_rbf_config.json
├── xgboost.joblib
├── xgboost_config.json
└── deep/
    ├── crnn_best.pt
    └── crnn_config.json
```

Each `*_config.json` stores `val_macro_f1`, `best_params`, `class_names`, training metadata, and library versions.

---

## 10. Project Structure

```
.
├── app.py                          # EcoGuard Streamlit app
├── requirements.txt                # Python dependencies
├── forest_threat_sound_detection_nti.ipynb   # Full training notebook
│
├── models/                         # Trained model files (not in repo — download separately)
│   ├── *.joblib                    # Classical ML models
│   ├── *_config.json               # Hyperparameters & metadata
│   └── deep/
│       ├── crnn_best.pt            # CRNN weights (PyTorch)
│       └── crnn_config.json
│
├── data/                           # (generated by notebook)
│   ├── features.parquet            # Pre-computed classical features
│   ├── train_manifest.csv
│   ├── val_manifest.csv
│   └── test_manifest.csv           # SHA-256 tripwire applied
│
├── samples/                        # Optional: real audio samples for the demo tab
│   └── *.wav
│
└── experiments_log.csv             # (generated by notebook) all tuning runs
```

---

## 11. Installation & Quick Start

### Prerequisites

- Python 3.9 or higher
- `ffmpeg` installed system-wide for MP3/OGG decoding  
  (`sudo apt install ffmpeg` / `brew install ffmpeg`)

### 1 — Clone the repository

```bash
git clone https://github.com/<your-org>/ecoguard-forest-sound-detection.git
cd ecoguard-forest-sound-detection
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

To also enable the CRNN deep-learning tab, install PyTorch:

```bash
# CPU-only (recommended for most users)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3 — Place trained models

Download or copy the trained model files into the `models/` folder (see [Project Structure](#10-project-structure)). The app will display exactly which models it loaded in the sidebar.

### 4 — Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### 5 — (Optional) Retrain from scratch

Open and run the notebook end-to-end:

```bash
jupyter notebook "forest_threat_sound_detection_nti.ipynb"
```

The notebook downloads the Kaggle datasets, builds `features.parquet`, trains and tunes all models, and exports `.joblib` + `_config.json` files into `models/`.

---

## 12. Usage Guide

### Audio input options

| Option | How |
|---|---|
| **Upload file** | Drop a WAV, MP3, or OGG recording |
| **Record live** | Uses your browser microphone (Streamlit ≥ 1.39) |
| **Sample / test signal** | Pick a file from `samples/` or choose a synthetic test signal |

### Sidebar controls

| Control | Purpose |
|---|---|
| **Max 3-second windows** | Limit how many consecutive windows are analysed (1–40) |
| **Model folder** | Override the path to `*.joblib` files (default: `models/`) |
| **Include CRNN** | Toggle to load/unload the deep-learning model |
| **Model for single prediction** | Which model to show in Tab 1 |
| **Classes treated as threats** | Select which of the 7 classes trigger an alert |
| **Threat alert threshold** | Probability threshold (0.05–0.99) for raising an alarm |

### Reading the alert

- 🔴 **RED pulsing box** — threat detected: shows the class, peak probability, and which window triggered it.
- 🟢 **GREEN pulsing box** — no threat: shows the best-matching benign class and its confidence.
- 🟠 **Orange near-miss** — peak threat probability is within 60% of the threshold; worth manual review.

---

## 13. Reproducibility

| Item | How it is enforced |
|---|---|
| Random seed | `seed=42` everywhere (sklearn, NumPy, PyTorch) |
| Data split | `StratifiedGroupKFold` with `group_id` — no clip leakage |
| Test integrity | SHA-256 tripwire on `test_manifest.csv` |
| CV scope | `GroupKFold` for inner loops; standard KFold is prohibited |
| Model selection | Fold 1 (val) macro-F1 only; test set is touched once per developer |
| Library versions | Pinned in `requirements.txt` (scikit-learn 1.6.1, LightGBM 4.6.0, XGBoost 3.2.0, CatBoost 1.2.10) |
| Experiment log | All tuning runs saved to `experiments_log.csv` |

---

## 14. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.39 | Web dashboard + live microphone input |
| `librosa` | ≥ 0.10 | Audio loading, MFCC, mel-spectrogram, spectral features |
| `soundfile` | latest | WAV read/write |
| `numpy` | latest | Numerical operations |
| `pandas` | latest | Feature dataframes, manifests |
| `plotly` | latest | Interactive charts in the app |
| `matplotlib` | latest | PNG spectrogram export |
| `joblib` | latest | Model serialisation / deserialisation |
| `scikit-learn` | 1.6.1 | Pipelines, scalers, LogReg, RF, evaluation metrics |
| `lightgbm` | 4.6.0 | Gradient boosting (Dev A) |
| `xgboost` | 3.2.0 | Gradient boosting (Dev B) |
| `catboost` | 1.2.10 | Gradient boosting (Dev B) |
| `torch` *(optional)* | any | CRNN deep-learning model (Tab 4) |

---

## License

This project is released under the [MIT License](LICENSE).

---

*EcoGuard · Forest Threat Sound Detection · 7-class acoustic classifier · 6 classical ML models + CNN-BiGRU deep learning*
