import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

print("Loading data and creating scalers...")

# Configuration (copy from src/config.py)
SEQUENCE_LENGTH = 24
TARGET_COL = "SM_20cm"
DATA_PATH = "data/final_sk4.csv"

# Load the data
df = pd.read_csv(DATA_PATH, parse_dates=['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)
df = df.interpolate(method='linear', limit_direction='both')

# Define the column groups
soil_cols = ['SM_0_5cm', 'ST_0_5cm', 'SM_5cm', 'ST_5cm', 'SM_20cm', 'ST_20cm',
             'SM_50cm', 'ST_50cm', 'SM_100cm', 'ST_100cm', 'SM_150cm', 'ST_150cm']
meteo_cols = ['Air_Temp', 'Precip_Total', 'Wind_Speed', 'Rel_Humidity',
              'Precip_24h_sum', 'Air_Temp_24h_mean', 'Wind_Speed_24h_mean',
              'Rel_Humidity_24h_mean', 'day_of_year_sin', 'day_of_year_cos',
              'hour_sin', 'hour_cos', 'is_frozen', 'Wind_Dir_sin', 'Wind_Dir_cos']

# Remove target from soil_cols for input features
input_soil_cols = [col for col in soil_cols if col != TARGET_COL]

print(f"Input soil features: {len(input_soil_cols)} columns")
print(f"Meteo features: {len(meteo_cols)} columns")

# Extract features
X_soil = df[input_soil_cols].values
X_meteo = df[meteo_cols].values
y = df[[TARGET_COL]].values

# Create and fit scalers
scaler_soil = StandardScaler()
scaler_meteo = StandardScaler()
scaler_y = StandardScaler()

scaler_soil.fit(X_soil)
scaler_meteo.fit(X_meteo)
scaler_y.fit(y)

# Save scalers
os.makedirs("models", exist_ok=True)

joblib.dump(scaler_soil, "models/scaler_soil.pkl")
joblib.dump(scaler_meteo, "models/scaler_meteo.pkl")
joblib.dump(scaler_y, "models/scaler_y.pkl")

print("✅ Scalers saved successfully to models/")

# Quick verification
print("\nVerifying scalers:")
print(f"scaler_soil mean shape: {scaler_soil.mean_.shape}")
print(f"scaler_meteo mean shape: {scaler_meteo.mean_.shape}")
print(f"scaler_y mean shape: {scaler_y.mean_.shape}")