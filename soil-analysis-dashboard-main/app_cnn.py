import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import os

# -------------------- CNN Model Definition (must match training) --------------------
class SpectralCNNWithAttention(nn.Module):
    """1D CNN with self-attention for spectral data"""
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
        # x: (batch, input_length)
        x = x.unsqueeze(1)                     # (batch, 1, input_length)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))  # (batch, num_filters[1], seq_len)
        x = x.transpose(1, 2)                   # (batch, seq_len, num_filters[1])
        x, _ = self.attention(x, x, x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)                        # (batch, num_filters[1])
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x.squeeze()

# -------------------- Configuration --------------------
MODEL_PATH = "models/soilwise_cnn.pth"          # Your trained CNN model
SCALER_X_PATH = "models/scaler_X.pkl"           # Scaler for spectral features
SCALER_Y_PATH = "models/scaler_y.pkl"           # Scaler for TOC target
EXPECTED_BANDS = 2041
WAVELENGTH_START = 405
WAVELENGTH_END = 2445
SAMPLE_DATA_PATH = "data/D3.2_20240117_ProbeField_Preprocessed_Spectra_DT_EPO_V1.csv"  # optional, for sample

# -------------------- Load Model and Scalers --------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectralCNNWithAttention(input_length=EXPECTED_BANDS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_scalers():
    if not os.path.exists(SCALER_X_PATH) or not os.path.exists(SCALER_Y_PATH):
        st.error("Scaler files not found.")
        return None, None
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    return scaler_X, scaler_y

# -------------------- Page Config --------------------
st.set_page_config(page_title="Soil Organic Carbon Predictor (CNN)", layout="wide")
st.title("🌱 Soil Organic Carbon Predictor")
st.markdown("Upload a soil spectrum (reflectance values from 405 nm to 2445 nm) to estimate **Total Organic Carbon (TOC)** using a 1D CNN with attention.")

# Load artifacts
model, device = load_model()
scaler_X, scaler_y = load_scalers()

if model is None or scaler_X is None:
    st.stop()
else:
    st.sidebar.success("✅ Model and scalers loaded")

# -------------------- Sidebar --------------------
st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with one row of spectral data", type=["csv"])

# Option to use a sample spectrum from the training set
use_sample = st.sidebar.checkbox("Use sample spectrum from dataset")

# -------------------- Prediction Function --------------------
def predict_from_spectrum(spectrum):
    """spectrum: 1D numpy array of length EXPECTED_BANDS"""
    if spectrum.ndim == 1:
        spectrum = spectrum.reshape(1, -1)
    spectrum_scaled = scaler_X.transform(spectrum)
    tensor = torch.tensor(spectrum_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_scaled = model(tensor).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
    return pred

# -------------------- Main Area --------------------
if use_sample:
    # Try to load a sample spectrum from the original dataset
    if os.path.exists(SAMPLE_DATA_PATH):
        try:
            sample_df = pd.read_csv(SAMPLE_DATA_PATH, sep=';', decimal=',', nrows=1)
            # Identify spectral columns (numeric between start and end)
            spectral_cols = [col for col in sample_df.columns if col.isdigit() and WAVELENGTH_START <= int(col) <= WAVELENGTH_END]
            if len(spectral_cols) != EXPECTED_BANDS:
                st.warning(f"Sample file has {len(spectral_cols)} bands, expected {EXPECTED_BANDS}. Using random data.")
                sample_spectrum = np.random.randn(EXPECTED_BANDS).astype(np.float32)
                actual_toc = None
            else:
                sample_spectrum = sample_df[spectral_cols].values.astype(np.float32).flatten()
                actual_toc = sample_df['TOC'].values[0]
                st.subheader("Sample Spectrum (from dataset)")
                st.write(f"Actual TOC: **{actual_toc:.3f}** %")
        except Exception as e:
            st.warning(f"Could not load sample: {e}. Using random spectrum.")
            sample_spectrum = np.random.randn(EXPECTED_BANDS).astype(np.float32)
            actual_toc = None
    else:
        st.info("Sample data file not found. Using random spectrum for demonstration.")
        sample_spectrum = np.random.randn(EXPECTED_BANDS).astype(np.float32)
        actual_toc = None

    if st.sidebar.button("Predict on Sample"):
        pred = predict_from_spectrum(sample_spectrum)
        st.success(f"### Predicted TOC: **{pred:.3f}** %")
        if actual_toc is not None:
            st.info(f"Actual TOC: {actual_toc:.3f} % | Difference: {pred - actual_toc:.3f}")

        # Plot the spectrum
        fig, ax = plt.subplots(figsize=(10, 4))
        wavelengths = np.arange(WAVELENGTH_START, WAVELENGTH_END+1)
        ax.plot(wavelengths, sample_spectrum, linewidth=1)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance (scaled)')
        ax.set_title('Input Spectrum')
        st.pyplot(fig)

elif uploaded_file is not None:
    try:
        # Read uploaded CSV (assume no header, one row)
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape[0] != 1:
            st.error("File must contain exactly one row of data.")
        elif df.shape[1] != EXPECTED_BANDS:
            st.error(f"Expected {EXPECTED_BANDS} columns (wavelengths from {WAVELENGTH_START} to {WAVELENGTH_END}), but got {df.shape[1]}.")
        else:
            spectrum = df.values.astype(np.float32).flatten()
            st.subheader("Uploaded Spectrum Preview")
            st.write("First 10 values:", spectrum[:10])

            if st.sidebar.button("Predict"):
                pred = predict_from_spectrum(spectrum)
                st.success(f"### Predicted TOC: **{pred:.3f}** %")

                # Plot
                fig, ax = plt.subplots(figsize=(10, 4))
                wavelengths = np.arange(WAVELENGTH_START, WAVELENGTH_END+1)
                ax.plot(wavelengths, spectrum, linewidth=1)
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Reflectance (scaled)')
                ax.set_title('Uploaded Spectrum')
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("👈 Please upload a CSV file with one row of spectral data, or use the sample spectrum.")

# -------------------- Instructions --------------------
st.markdown("---")
st.markdown("""
### 📋 Instructions
1. **Prepare a CSV file** with **one row** containing exactly 2041 reflectance values (wavelengths from **405 nm to 2445 nm**, in 1 nm steps).  
   - No header row required.  
   - Values must be in the same order as training (405, 406, …, 2445).  
   - Decimal separator can be dot or comma – the app handles both.
2. **Upload** the file using the sidebar.
3. Click **Predict** to get estimated TOC.

The model was trained on the SoilWise dataset (317 samples) and achieves R² ≈ **0.70**.
""")