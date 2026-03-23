import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("Reorganising model files...")

# 1. Regenerate soil moisture target scaler (if needed)
soil_data_path = "data/final_sk4.csv"
if os.path.exists(soil_data_path):
    print("Regenerating soil moisture target scaler...")
    df = pd.read_csv(soil_data_path, parse_dates=['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df = df.interpolate(method='linear', limit_direction='both')
    y = df[['SM_20cm']].values
    scaler_y_soil = StandardScaler()
    scaler_y_soil.fit(y)
    joblib.dump(scaler_y_soil, "models/scaler_y_soil.pkl")
    print("Saved models/scaler_y_soil.pkl")
else:
    print("Soil data not found, skipping soil scaler regeneration.")

# 2. Create cnn subfolder
cnn_dir = "models/cnn"
os.makedirs(cnn_dir, exist_ok=True)

# 3. Move CNN-related files if they exist
cnn_files = ["scaler_X.pkl", "scaler_y.pkl", "soilwise_cnn.pth"]
for f in cnn_files:
    src = os.path.join("models", f)
    dst = os.path.join(cnn_dir, f)
    if os.path.exists(src):
        os.rename(src, dst)
        print(f"Moved {src} -> {dst}")
    else:
        print(f"{src} not found, skipping.")

print("Reorganisation complete.")