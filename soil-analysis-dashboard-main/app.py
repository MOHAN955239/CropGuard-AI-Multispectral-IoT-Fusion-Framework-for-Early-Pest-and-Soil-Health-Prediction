import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from src.model import FusionModel
from src.config import SEQUENCE_LENGTH, DEVICE

# Page config
st.set_page_config(page_title="Soil Moisture Predictor", layout="wide")
st.title("🌱 Soil Moisture Prediction Dashboard")
st.markdown("Predict soil moisture at **20 cm depth** for the next time step using multi‑modal sensor data.")

# Define column groups - EXCLUDE target from input features
soil_cols = ['SM_0_5cm', 'ST_0_5cm', 'SM_5cm', 'ST_5cm', 'ST_20cm',  # SM_20cm is target, excluded
             'SM_50cm', 'ST_50cm', 'SM_100cm', 'ST_100cm', 'SM_150cm', 'ST_150cm']
meteo_cols = ['Air_Temp', 'Precip_Total', 'Wind_Speed', 'Rel_Humidity',
              'Precip_24h_sum', 'Air_Temp_24h_mean', 'Wind_Speed_24h_mean',
              'Rel_Humidity_24h_mean', 'day_of_year_sin', 'day_of_year_cos',
              'hour_sin', 'hour_cos', 'is_frozen', 'Wind_Dir_sin', 'Wind_Dir_cos']
target_col = 'SM_20cm'

@st.cache_resource
def load_artifacts():
    # Load model
    model = FusionModel(input_dim1=len(soil_cols), input_dim2=len(meteo_cols))
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Load scalers
    scaler_soil = joblib.load("models/scaler_soil.pkl")
    scaler_meteo = joblib.load("models/scaler_meteo.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")
    
    return model, scaler_soil, scaler_meteo, scaler_y

# Load artifacts
try:
    model, scaler_soil, scaler_meteo, scaler_y = load_artifacts()
    st.sidebar.success("✅ Model and scalers loaded")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Sidebar controls
st.sidebar.header("Input Data")
option = st.sidebar.radio("Choose input method:", ("Upload CSV", "Use sample data"))

# Y‑axis scaling option
st.sidebar.header("Plot Settings")
use_global_y = st.sidebar.checkbox("Use global y‑axis range", value=True,
                                   help="If checked, y‑axis spans the full training range (0.3σ around mean). "
                                        "Otherwise, it zooms to the data + prediction.")
# Compute global y‑limits from training target distribution
global_y_min = scaler_y.mean_[0] - 3 * scaler_y.scale_[0]
global_y_max = scaler_y.mean_[0] + 3 * scaler_y.scale_[0]

def predict_from_df(df):
    """Take a DataFrame with at least SEQUENCE_LENGTH rows, preprocess, predict."""
    if len(df) < SEQUENCE_LENGTH:
        st.error(f"Need at least {SEQUENCE_LENGTH} rows. Uploaded {len(df)}.")
        return None, None, False
    
    # Use last SEQUENCE_LENGTH rows
    df_seq = df.iloc[-SEQUENCE_LENGTH:].copy()
    
    # Check required columns
    missing_soil = [c for c in soil_cols if c not in df_seq.columns]
    missing_meteo = [c for c in meteo_cols if c not in df_seq.columns]
    if missing_soil or missing_meteo:
        st.error(f"Missing columns. Soil: {missing_soil}, Meteo: {missing_meteo}")
        return None, None, False
    
    # Check if target column exists for plotting
    has_target = target_col in df_seq.columns
    
    # Extract and scale input features only
    X_soil = scaler_soil.transform(df_seq[soil_cols].values)
    X_meteo = scaler_meteo.transform(df_seq[meteo_cols].values)
    
    # Convert to tensor (batch_size=1, seq_len, features)
    x1 = torch.FloatTensor(X_soil).unsqueeze(0).to(DEVICE)
    x2 = torch.FloatTensor(X_meteo).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        pred_scaled, _ = model(x1, x2)
    pred = scaler_y.inverse_transform(pred_scaled.cpu().numpy().reshape(-1, 1))[0, 0]
    
    return pred, df_seq, has_target

# Plotting function with flexible y‑axis
def plot_results(df_seq, pred, has_target):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Prepare time axis
    if 'Timestamp' in df_seq.columns:
        time_index = pd.to_datetime(df_seq['Timestamp'])
        next_time = time_index.iloc[-1] + pd.Timedelta(hours=1)
        # Extend x‑axis to include prediction time
        all_times = list(time_index) + [next_time]
        ax.set_xlim([min(all_times), max(all_times)])
    else:
        time_index = np.arange(len(df_seq))
        next_time = len(df_seq)
        ax.set_xlim([0, next_time + 0.5])
    
    # Plot past data if available
    if has_target:
        ax.plot(time_index, df_seq[target_col], 'o-', label='Actual (past)')
    
    # Set y‑limits
    if use_global_y:
        ax.set_ylim([global_y_min, global_y_max])
    else:
        # Dynamic limits based on data + prediction
        y_values = df_seq[target_col].tolist() if has_target else []
        y_values.append(pred)
        y_min, y_max = min(y_values), max(y_values)
        y_range = y_max - y_min
        padding = y_range * 0.1 if y_range > 0 else 0.1
        ax.set_ylim([y_min - padding, y_max + padding])
    
    # Plot prediction
    ax.scatter([next_time], [pred], color='red', s=100, label='Predicted (next)')
    ax.set_ylabel('Soil Moisture (m³/m³)')
    ax.set_title(f'{target_col}: Past {SEQUENCE_LENGTH} steps + Prediction')
    ax.legend()
    st.pyplot(fig)

if option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Try to parse Timestamp if present
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())
        if st.sidebar.button("Predict"):
            pred, df_seq, has_target = predict_from_df(df)
            if pred is not None:
                st.success(f"### Predicted soil moisture at 20 cm: **{pred:.4f} m³/m³**")
                plot_results(df_seq, pred, has_target)

else:  # Use sample data
    st.sidebar.info("Using a sample sequence from the dataset.")
    try:
        df_full = pd.read_csv("data/final_sk4.csv", parse_dates=['Timestamp'])
        # Pick a random continuous sequence of length SEQUENCE_LENGTH
        max_start = len(df_full) - SEQUENCE_LENGTH
        start = np.random.randint(0, max_start)
        sample_df = df_full.iloc[start:start+SEQUENCE_LENGTH].copy()
        st.subheader("Sample Input Data")
        st.dataframe(sample_df)
        
        if st.sidebar.button("Run Prediction on Sample"):
            pred, df_seq, has_target = predict_from_df(sample_df)
            if pred is not None:
                st.success(f"### Predicted soil moisture at 20 cm: **{pred:.4f} m³/m³**")
                plot_results(df_seq, pred, has_target)
    except FileNotFoundError:
        st.error("Sample data not available. Please upload a CSV file.")