import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# -------------------- Configuration --------------------
DATA_FILE = r"D:\Crop Final\data\D3.2_20240117_ProbeField_Preprosessed_Spectra_DT_EPO_V1.csv"
TARGET_COL = 'TOC'
SPECTRAL_START = 405
SPECTRAL_END = 2445
MODEL_PATH = "models/cnn/soilwise_cnn.pth"
SCALER_X_PATH = "models/cnn/scaler_X.pkl"
SCALER_Y_PATH = "models/cnn/scaler_y.pkl"
RANDOM_SEED = 42
TEST_SIZE = 0.2

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

# -------------------- Load Data --------------------
print("Loading data...")
df = pd.read_csv(DATA_FILE, sep=';', decimal=',', low_memory=False)

# Identify spectral columns
spectral_cols = [c for c in df.columns if c.isdigit() and SPECTRAL_START <= int(c) <= SPECTRAL_END]
print(f"Found {len(spectral_cols)} spectral bands.")

# Drop rows with missing TOC
df = df.dropna(subset=[TARGET_COL]).copy()
print(f"Total samples: {len(df)}")

# Convert TOC to numeric (in case it's string)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

X = df[spectral_cols].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32)

# -------------------- Split into train/test --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# -------------------- Load scalers and scale test data --------------------
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

X_test_scaled = scaler_X.transform(X_test)

# -------------------- Load model --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpectralCNNWithAttention(input_length=len(spectral_cols))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -------------------- Predict on test set --------------------
print("\nRunning predictions on test set...")
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_pred_scaled = model(X_test_tensor).cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# -------------------- Metrics --------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*50)
print("CNN MODEL EVALUATION")
print("="*50)
print(f"Test set size: {len(y_test)} samples")
print(f"R²  = {r2:.4f}")
print(f"MAE = {mae:.4f} %")
print(f"RMSE= {rmse:.4f} %")
print("="*50)

# Optional: Plot predicted vs actual
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual TOC (%)')
    plt.ylabel('Predicted TOC (%)')
    plt.title('CNN: Predicted vs Actual TOC')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/cnn_test_scatter.png')
    print("Scatter plot saved to results/cnn_test_scatter.png")
except:
    print("Matplotlib not available, skipping plot.")