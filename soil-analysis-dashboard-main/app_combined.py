import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import librosa.display
import pickle
import plotly.graph_objects as go
import plotly.express as px
from src.model import FusionModel
from src.config import SEQUENCE_LENGTH, DEVICE

# ==================== Get absolute path of script directory ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== Page Config ====================
st.set_page_config(page_title="Soil Analysis Dashboard", layout="wide")
st.title("🌱 Soil Analysis Dashboard")
st.markdown("Predict **soil moisture**, **Total Organic Carbon**, **Fertility Score**, or **Pest Sounds**.")

# ==================== Multi‑modal Model Definitions ====================
soil_cols = ['SM_0_5cm', 'ST_0_5cm', 'SM_5cm', 'ST_5cm', 'ST_20cm',
             'SM_50cm', 'ST_50cm', 'SM_100cm', 'ST_100cm', 'SM_150cm', 'ST_150cm']
meteo_cols = ['Air_Temp', 'Precip_Total', 'Wind_Speed', 'Rel_Humidity',
              'Precip_24h_sum', 'Air_Temp_24h_mean', 'Wind_Speed_24h_mean',
              'Rel_Humidity_24h_mean', 'day_of_year_sin', 'day_of_year_cos',
              'hour_sin', 'hour_cos', 'is_frozen', 'Wind_Dir_sin', 'Wind_Dir_cos']
target_col_mm = 'SM_20cm'

MM_MODEL_PATH = os.path.join(SCRIPT_DIR, "models/best_model.pth")
MM_SCALER_SOIL = os.path.join(SCRIPT_DIR, "models/scaler_soil.pkl")
MM_SCALER_METEO = os.path.join(SCRIPT_DIR, "models/scaler_meteo.pkl")
MM_SCALER_Y = os.path.join(SCRIPT_DIR, "models/scaler_y_soil.pkl")

# ==================== CNN Model Definitions ====================
class SpectralCNNWithAttention(nn.Module):
    def __init__(self, input_length=2041, num_filters=[64, 128], kernel_size=5, num_heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, num_filters[0], kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(num_filters[0], num_filters[1], kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_filters[1])
        self.pool2 = nn.MaxPool1d(2)

        self.attention = nn.MultiheadAttention(embed_dim=num_filters[1], num_heads=num_heads, batch_first=True, dropout=dropout)

        self.fc1 = nn.Linear(num_filters[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(num_filters[1])

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        x, _ = self.attention(x, x, x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x.squeeze()

CNN_MODEL_PATH = os.path.join(SCRIPT_DIR, "models/cnn/soilwise_cnn.pth")
CNN_SCALER_X = os.path.join(SCRIPT_DIR, "models/cnn/scaler_X.pkl")
CNN_SCALER_Y = os.path.join(SCRIPT_DIR, "models/cnn/scaler_y.pkl")
EXPECTED_BANDS = 2041
WAVELENGTH_START = 405
WAVELENGTH_END = 2445
SAMPLE_DATA_PATH = r"D:\Crop Final\data\D3.2_20240117_ProbeField_Preprosessed_Spectra_DT_EPO_V1.csv"

# ==================== Fertility Model Definitions ====================
class FertilityMLP(nn.Module):
    def __init__(self, input_dim=12, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

FERTILITY_MODEL_PATH = os.path.join(SCRIPT_DIR, "models/mlp_model_improved/fertility_model_final.pth")
FERTILITY_SCALER_PATH = os.path.join(SCRIPT_DIR, "models/mlp_model_improved/scaler.pkl")
FERTILITY_FEATURES = ['N', 'P', 'K', 'ph', 'ec', 'oc', 'S', 'zn', 'fe', 'cu', 'Mn', 'B']
FERTILITY_DATA_PATH = r"D:\Crop Final\data\Soil Fertility Data (Modified Data).csv"

# ==================== Audio Model Definitions (Pest Sound) ====================
AUDIO_MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "audio_model")
AUDIO_ENCODER_PATH = os.path.join(AUDIO_MODEL_DIR, "label_encoder.pkl")

# Prioritise .h5 files (directly loadable by Keras 3)
AUDIO_MODEL_CANDIDATES = [
    "final_model.h5",
    "best_model.h5",
    "final_model.keras.zip",
    "best_model.keras.zip",
    "final_model.keras",
    "best_model.keras"
]

# Audio feature extraction parameters
AUDIO_SAMPLE_RATE = 22050      # Hz
AUDIO_DURATION = 5.0            # seconds
AUDIO_N_MELS = 128
AUDIO_N_FFT = 2048
AUDIO_HOP_LENGTH = 512
AUDIO_N_CHANNELS = 1

# ==================== Load Models (cached) ====================
@st.cache_resource
def load_multimodal():
    model = FusionModel(input_dim1=len(soil_cols), input_dim2=len(meteo_cols))
    model.load_state_dict(torch.load(MM_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    scaler_soil = joblib.load(MM_SCALER_SOIL)
    scaler_meteo = joblib.load(MM_SCALER_METEO)
    scaler_y = joblib.load(MM_SCALER_Y)
    return model, scaler_soil, scaler_meteo, scaler_y

@st.cache_resource
def load_cnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectralCNNWithAttention(input_length=EXPECTED_BANDS)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    scaler_X = joblib.load(CNN_SCALER_X)
    scaler_y = joblib.load(CNN_SCALER_Y)
    return model, scaler_X, scaler_y, device

@st.cache_resource
def load_fertility():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FertilityMLP()
    model.load_state_dict(torch.load(FERTILITY_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    scaler = joblib.load(FERTILITY_SCALER_PATH)
    return model, scaler, device

@st.cache_resource
def load_audio_model():
    """Load Keras CRNN model and label encoder for pest sound classification."""
    if not os.path.exists(AUDIO_MODEL_DIR):
        st.sidebar.error(f"Audio model directory not found: {AUDIO_MODEL_DIR}")
        return None, None

    # Try each candidate silently
    model_path = None
    last_error = None
    for candidate in AUDIO_MODEL_CANDIDATES:
        full_path = os.path.join(AUDIO_MODEL_DIR, candidate)
        if os.path.exists(full_path):
            try:
                model = tf.keras.models.load_model(full_path)
                # Success – break out
                model_path = full_path
                break
            except Exception as e:
                last_error = e
                continue

    if model_path is None:
        st.sidebar.error(f"No loadable model found in {AUDIO_MODEL_DIR}. Last error: {last_error}")
        return None, None

    # Load encoder
    try:
        with open(AUDIO_ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
    except Exception as e:
        st.sidebar.error(f"Error loading label encoder: {e}")
        return None, None

    st.sidebar.success(f"✅ Audio model loaded")
    return model, encoder

# ==================== Load dataset and compute class medians for fertility sample buttons ====================
@st.cache_data
def get_fertility_class_medians():
    df = pd.read_csv(FERTILITY_DATA_PATH)
    class_medians = {}
    for cls in [0, 1, 2]:
        subset = df[df['fertility'] == cls]
        if len(subset) > 0:
            med = subset[FERTILITY_FEATURES].median().to_dict()
            class_medians[cls] = med
        else:
            class_medians[cls] = {f: 0 for f in FERTILITY_FEATURES}
    return class_medians

# ==================== Feature extraction for audio (pest sound) ====================
def extract_audio_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=AUDIO_SAMPLE_RATE, duration=AUDIO_DURATION)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=AUDIO_N_MELS,
            n_fft=AUDIO_N_FFT,
            hop_length=AUDIO_HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = mel_db.T  # (time, n_mels)
        features = mel_db[..., np.newaxis]          # (time, n_mels, 1)
        features = np.expand_dims(features, axis=0) # (1, time, n_mels, 1)
        return features
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

# ==================== Helper functions for audio tab ====================
def get_audio_info(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        return {
            'duration': duration,
            'sample_rate': sr,
            'file_size': os.path.getsize(audio_path) / 1024
        }
    except:
        return None

def plot_waveform(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(audio, sr=sr, ax=ax, color='#2E7D32')
        ax.set_title('Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        return fig
    except:
        return None

def plot_spectrogram(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, duration=AUDIO_DURATION)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=AUDIO_N_MELS,
            n_fft=AUDIO_N_FFT,
            hop_length=AUDIO_HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel_db, sr=sr, hop_length=AUDIO_HOP_LENGTH,
            x_axis='time', y_axis='mel', ax=ax
        )
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel-Spectrogram')
        return fig
    except:
        return None

def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#2E7D32"},
            'steps': [
                {'range': [0, 50], 'color': "#ffcccc"},
                {'range': [50, 75], 'color': "#ffffcc"},
                {'range': [75, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# ==================== Sidebar ====================
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio(
    "Choose model",
    ["Soil Moisture (Multi‑modal)", "Soil Organic Carbon (CNN)", 
     "Fertility Score (MLP)", "Pest Sound (CRNN)"]
)

# ==================== Multi‑modal Tab ====================
if app_mode == "Soil Moisture (Multi‑modal)":
    # ... (unchanged, same as before) ...
    st.header("🌧️ Soil Moisture Predictor (Multi‑modal GRU + Attention)")
    st.markdown("Predict soil moisture at **20 cm depth** for the next hour using sensor data.")

    try:
        mm_model, mm_scaler_soil, mm_scaler_meteo, mm_scaler_y = load_multimodal()
        st.sidebar.success("✅ Multi‑modal model loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading multi‑modal model: {e}")
        st.stop()

    global_y_min = mm_scaler_y.mean_[0] - 3 * mm_scaler_y.scale_[0]
    global_y_max = mm_scaler_y.mean_[0] + 3 * mm_scaler_y.scale_[0]

    st.sidebar.header("Input Data (Multi‑modal)")
    option = st.sidebar.radio("Choose input method:", ("Upload CSV", "Use sample data"))
    use_global_y = st.sidebar.checkbox("Use global y‑axis range", value=True)

    def predict_mm(df):
        if len(df) < SEQUENCE_LENGTH:
            st.error(f"Need at least {SEQUENCE_LENGTH} rows.")
            return None, None, False
        df_seq = df.iloc[-SEQUENCE_LENGTH:].copy()
        missing_soil = [c for c in soil_cols if c not in df_seq.columns]
        missing_meteo = [c for c in meteo_cols if c not in df_seq.columns]
        if missing_soil or missing_meteo:
            st.error(f"Missing columns. Soil: {missing_soil}, Meteo: {missing_meteo}")
            return None, None, False
        has_target = target_col_mm in df_seq.columns
        X_soil = mm_scaler_soil.transform(df_seq[soil_cols].values)
        X_meteo = mm_scaler_meteo.transform(df_seq[meteo_cols].values)
        x1 = torch.FloatTensor(X_soil).unsqueeze(0).to(DEVICE)
        x2 = torch.FloatTensor(X_meteo).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_scaled, _ = mm_model(x1, x2)
        pred = mm_scaler_y.inverse_transform(pred_scaled.cpu().numpy().reshape(-1, 1))[0, 0]
        return pred, df_seq, has_target

    def plot_mm(df_seq, pred, has_target):
        fig, ax = plt.subplots(figsize=(10, 4))
        if 'Timestamp' in df_seq.columns:
            time_index = pd.to_datetime(df_seq['Timestamp'])
            next_time = time_index.iloc[-1] + pd.Timedelta(hours=1)
            all_times = list(time_index) + [next_time]
            ax.set_xlim([min(all_times), max(all_times)])
        else:
            time_index = np.arange(len(df_seq))
            next_time = len(df_seq)
            ax.set_xlim([0, next_time + 0.5])
        if has_target:
            ax.plot(time_index, df_seq[target_col_mm], 'o-', label='Actual (past)')
        if use_global_y:
            ax.set_ylim([global_y_min, global_y_max])
        else:
            y_values = df_seq[target_col_mm].tolist() if has_target else []
            y_values.append(pred)
            y_min, y_max = min(y_values), max(y_values)
            padding = (y_max - y_min) * 0.1 or 0.1
            ax.set_ylim([y_min - padding, y_max + padding])
        ax.scatter([next_time], [pred], color='red', s=100, label='Predicted (next)')
        ax.set_ylabel('Soil Moisture (m³/m³)')
        ax.set_title(f'{target_col_mm}: Past {SEQUENCE_LENGTH} steps + Prediction')
        ax.legend()
        st.pyplot(fig)

    if option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="mm_upload")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head())
            if st.sidebar.button("Predict", key="mm_predict"):
                pred, df_seq, has_target = predict_mm(df)
                if pred is not None:
                    st.success(f"### Predicted soil moisture at 20 cm: **{pred:.4f} m³/m³**")
                    plot_mm(df_seq, pred, has_target)
    else:  # sample data
        try:
            df_full = pd.read_csv("data/final_sk4.csv", parse_dates=['Timestamp'])
            max_start = len(df_full) - SEQUENCE_LENGTH
            start = np.random.randint(0, max_start)
            sample_df = df_full.iloc[start:start+SEQUENCE_LENGTH].copy()
            st.subheader("Sample Input Data")
            st.dataframe(sample_df)
            if st.sidebar.button("Run Prediction on Sample", key="mm_sample"):
                pred, df_seq, has_target = predict_mm(sample_df)
                if pred is not None:
                    st.success(f"### Predicted soil moisture at 20 cm: **{pred:.4f} m³/m³**")
                    plot_mm(df_seq, pred, has_target)
        except FileNotFoundError:
            st.error("Sample data file not found.")

# ==================== CNN Tab ====================
elif app_mode == "Soil Organic Carbon (CNN)":
    # ... (unchanged) ...
    st.header("🌿 Soil Organic Carbon Predictor (CNN)")
    st.markdown("Upload a soil spectrum (reflectance values from 405 nm to 2445 nm) to estimate **Total Organic Carbon (TOC)**.")

    try:
        cnn_model, cnn_scaler_X, cnn_scaler_y, cnn_device = load_cnn()
        st.sidebar.success("✅ CNN model loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading CNN model: {e}")
        st.stop()

    st.sidebar.header("Input Data (CNN)")
    uploaded_file = st.sidebar.file_uploader("Upload CSV with one row", type=["csv"], key="cnn_upload")
    use_sample = st.sidebar.checkbox("Use sample spectrum from dataset")

    def predict_cnn(spectrum):
        if spectrum.ndim == 1:
            spectrum = spectrum.reshape(1, -1)
        spectrum_scaled = cnn_scaler_X.transform(spectrum)
        tensor = torch.tensor(spectrum_scaled, dtype=torch.float32).to(cnn_device)
        with torch.no_grad():
            pred_scaled = cnn_model(tensor).cpu().numpy()
        pred = cnn_scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        return pred

    if use_sample:
        try:
            sample_df = pd.read_csv(SAMPLE_DATA_PATH, sep=';', decimal=',', nrows=1)
            spectral_cols = [c for c in sample_df.columns if c.isdigit() and WAVELENGTH_START <= int(c) <= WAVELENGTH_END]
            if len(spectral_cols) != EXPECTED_BANDS:
                st.warning(f"Sample has {len(spectral_cols)} bands, expected {EXPECTED_BANDS}. Using random.")
                sample_spectrum = np.random.randn(EXPECTED_BANDS).astype(np.float32)
                actual_toc = None
            else:
                sample_spectrum = sample_df[spectral_cols].values.astype(np.float32).flatten()
                toc_val = sample_df['TOC'].values[0]
                try:
                    actual_toc = float(toc_val)
                except ValueError:
                    actual_toc = None
                    st.warning("TOC value in sample is not numeric.")
                else:
                    st.subheader("Sample Spectrum")
                    st.write(f"Actual TOC: **{actual_toc:.3f}** %")
        except Exception as e:
            st.warning(f"Could not load sample: {e}. Using random spectrum.")
            sample_spectrum = np.random.randn(EXPECTED_BANDS).astype(np.float32)
            actual_toc = None

        if st.sidebar.button("Predict on Sample", key="cnn_sample"):
            pred = predict_cnn(sample_spectrum)
            st.success(f"### Predicted TOC: **{pred:.3f}** %")
            if actual_toc is not None:
                st.info(f"Actual: {actual_toc:.3f} % | Difference: {pred - actual_toc:.3f}")
            fig, ax = plt.subplots(figsize=(10, 4))
            wavelengths = np.arange(WAVELENGTH_START, WAVELENGTH_END+1)
            ax.plot(wavelengths, sample_spectrum, linewidth=1, color='blue')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Reflectance (scaled)')
            ax.set_title('Input Spectrum')
            textstr = f'Predicted TOC: {pred:.3f}%'
            if actual_toc is not None:
                textstr += f'\nActual TOC: {actual_toc:.3f}%'
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            st.pyplot(fig)

    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            if df.shape[0] != 1:
                st.error("File must contain exactly one row.")
            elif df.shape[1] != EXPECTED_BANDS:
                st.error(f"Expected {EXPECTED_BANDS} columns, got {df.shape[1]}.")
            else:
                spectrum = df.values.astype(np.float32).flatten()
                st.subheader("Uploaded Spectrum Preview")
                st.write("First 10 values:", spectrum[:10])
                if st.sidebar.button("Predict", key="cnn_predict"):
                    pred = predict_cnn(spectrum)
                    st.success(f"### Predicted TOC: **{pred:.3f}** %")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    wavelengths = np.arange(WAVELENGTH_START, WAVELENGTH_END+1)
                    ax.plot(wavelengths, spectrum, linewidth=1, color='blue')
                    ax.set_xlabel('Wavelength (nm)')
                    ax.set_ylabel('Reflectance (scaled)')
                    ax.set_title('Uploaded Spectrum')
                    textstr = f'Predicted TOC: {pred:.3f}%'
                    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("👈 Upload a CSV file with one row of spectral data, or use the sample spectrum.")

# ==================== Fertility Tab ====================
elif app_mode == "Fertility Score (MLP)":
    # ... (unchanged) ...
    st.header("🌾 Soil Fertility Predictor (MLP)")
    st.markdown("Enter soil chemical properties to estimate a **fertility score** (0 = low, 1 = high).")

    try:
        fertility_model, fertility_scaler, fertility_device = load_fertility()
        st.sidebar.success("✅ Fertility model loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading fertility model: {e}")
        st.stop()

    class_medians = get_fertility_class_medians()
    DEFAULT_VALS = class_medians.get(1, {f: 0 for f in FERTILITY_FEATURES})
    SAMPLE_LOW = class_medians.get(0, DEFAULT_VALS)
    SAMPLE_MEDIUM = class_medians.get(1, DEFAULT_VALS)
    SAMPLE_HIGH = class_medians.get(2, DEFAULT_VALS)

    if 'fertility_vals' not in st.session_state:
        st.session_state.fertility_vals = DEFAULT_VALS.copy()

    st.sidebar.header("Input Values")
    cols = st.sidebar.columns(3)
    for i, feature in enumerate(FERTILITY_FEATURES):
        with cols[i % 3]:
            current_val = st.session_state.fertility_vals[feature]
            new_val = st.number_input(f"{feature}", value=current_val, format="%.2f")
            st.session_state.fertility_vals[feature] = new_val

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Load sample values**")

    def set_sample(sample_dict):
        st.session_state.fertility_vals = sample_dict.copy()
        st.rerun()

    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("🌱 Low", use_container_width=True):
            set_sample(SAMPLE_LOW)
    with col2:
        if st.button("🌿 Medium", use_container_width=True):
            set_sample(SAMPLE_MEDIUM)
    with col3:
        if st.button("🌳 High", use_container_width=True):
            set_sample(SAMPLE_HIGH)

    if st.sidebar.button("Predict Fertility", key="fertility_predict", type="primary"):
        input_values = [st.session_state.fertility_vals[f] for f in FERTILITY_FEATURES]
        input_array = np.array(input_values, dtype=np.float32).reshape(1, -1)
        input_scaled = fertility_scaler.transform(input_array)

        print("\n--- Fertility Prediction ---")
        print("Raw inputs :", input_values)
        print("Scaled     :", input_scaled.flatten())

        tensor = torch.tensor(input_scaled, dtype=torch.float32).to(fertility_device)

        with torch.no_grad():
            raw_output = fertility_model(tensor).cpu().numpy()[0, 0]
            print("Raw model output:", raw_output)

        pred_scaled = float(raw_output)

        if pred_scaled < 0.25:
            fertility_class = "Low"
            color = "🔴"
        elif pred_scaled < 0.75:
            fertility_class = "Medium"
            color = "🟡"
        else:
            fertility_class = "High"
            color = "🟢"

        st.success(f"### Predicted Fertility Score: **{pred_scaled:.3f}**")
        st.info(f"**Fertility Class:** {color} {fertility_class}")

        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh([0], [pred_scaled], color='green', height=0.5)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.axvline(0.25, color='orange', linestyle='--', alpha=0.7)
        ax.axvline(0.75, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Fertility Score')
        ax.set_title('Soil Fertility Gauge')
        st.pyplot(fig)

        imp_path = "models/mlp_model_improved/feature_importance.png"
        if os.path.exists(imp_path):
            st.image(imp_path, caption="Feature Importance (higher = more important)")

# ==================== Pest Sound (CRNN) Tab ====================
else:
    st.header("🐞 Pest Sound Classification (CRNN)")
    st.markdown("Upload an audio file of an insect sound to identify the pest species.")

    # Load audio model (cached)
    audio_model, audio_encoder = load_audio_model()
    if audio_model is None or audio_encoder is None:
        st.sidebar.error("❌ Audio model not loaded. Check files in audio_model/")
        st.stop()
    # else: success message already shown in loader

    # Single prediction UI
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose a WAV/MP3 file",
            type=['wav', 'mp3', 'ogg', 'm4a'],
            key="audio_upload"
        )

        if uploaded_file is not None:
            # Save temporarily
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Display audio info
            audio_info = get_audio_info(temp_path)
            if audio_info:
                st.markdown("**Audio Information:**")
                st.json({
                    "Duration": f"{audio_info['duration']:.2f} s",
                    "Sample Rate": f"{audio_info['sample_rate']} Hz",
                    "File Size": f"{audio_info['file_size']:.1f} KB"
                })

            # Audio player
            st.audio(temp_path)

            # Optionally show waveform and spectrogram
            with st.expander("Show waveform & spectrogram"):
                fig_wave = plot_waveform(temp_path)
                if fig_wave:
                    st.pyplot(fig_wave)
                fig_spec = plot_spectrogram(temp_path)
                if fig_spec:
                    st.pyplot(fig_spec)

    with col2:
        if uploaded_file is not None:
            st.subheader("Prediction Results")

            with st.spinner("Analyzing audio..."):
                # Extract features
                features = extract_audio_features(temp_path)
                if features is not None:
                    # Predict
                    predictions = audio_model.predict(features, verbose=0)[0]

                    # Get top 5 predictions
                    top_k = 5
                    top_indices = np.argsort(predictions)[-top_k:][::-1]

                    # Main prediction
                    main_idx = top_indices[0]
                    main_pest = audio_encoder.inverse_transform([main_idx])[0]
                    main_confidence = predictions[main_idx]

                    # Display main prediction
                    st.markdown("### 🎯 Primary Prediction")
                    st.markdown(f"#### **{main_pest}**")
                    conf_pct = main_confidence * 100
                    if main_confidence > 0.8:
                        conf_class = "confidence-high"
                    elif main_confidence > 0.5:
                        conf_class = "confidence-medium"
                    else:
                        conf_class = "confidence-low"
                    st.markdown(f"##### Confidence: <span class='{conf_class}'>{conf_pct:.2f}%</span>", unsafe_allow_html=True)

                    # Gauge chart
                    gauge_fig = create_confidence_gauge(main_confidence)
                    st.plotly_chart(gauge_fig, use_container_width=True)

                    # Top K predictions
                    st.subheader(f"📊 Top {top_k} Predictions")
                    pred_data = []
                    for i, idx in enumerate(top_indices, 1):
                        pest = audio_encoder.inverse_transform([idx])[0]
                        conf = predictions[idx]
                        pred_data.append({
                            "Rank": i,
                            "Pest Species": pest,
                            "Confidence": f"{conf*100:.2f}%",
                            "Probability": conf
                        })
                    pred_df = pd.DataFrame(pred_data)

                    # Bar chart
                    fig = px.bar(
                        pred_df,
                        x="Confidence",
                        y="Pest Species",
                        orientation='h',
                        color="Confidence",
                        color_continuous_scale=["#ffcccc", "#ffffcc", "#ccffcc"],
                        title="Prediction Probabilities"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # Table
                    st.dataframe(pred_df[["Rank", "Pest Species", "Confidence"]], use_container_width=True)
                else:
                    st.error("Failed to extract features from audio")

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        else:
            st.info("👆 Upload an audio file to get started")

# ==================== Footer ====================
st.markdown("---")
st.markdown("""
### 📋 Instructions
- **Soil Moisture (Multi‑modal)**: Upload a CSV with at least 24 rows of sensor data. Required columns are listed in the app.
- **Soil Organic Carbon (CNN)**: Upload a CSV with a single row containing exactly 2041 reflectance values (wavelengths 405–2445 nm). No header.
- **Fertility Score (MLP)**: Enter the 12 soil chemical properties manually, or load one of the three sample profiles.
- **Pest Sound (CRNN)**: Upload an audio file (WAV, MP3, etc.) to identify the pest species.
""")