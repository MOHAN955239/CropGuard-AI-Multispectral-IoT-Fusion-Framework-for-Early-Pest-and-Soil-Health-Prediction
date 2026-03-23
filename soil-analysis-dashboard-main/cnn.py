import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time

# -------------------- Configuration --------------------
DATA_FILE = r'D:\Crop Final\data\D3.2_20240117_ProbeField_Preprosessed_Spectra_DT_EPO_V1.csv'
TARGET_COL = 'TOC'                     # from the column list
SPECTRAL_START = 405                    # first wavelength column
SPECTRAL_END = 2445                     # last wavelength column
N_FOLDS = 5
RANDOM_SEED = 42
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

# Choose which models to run
RUN_LINEAR_REGRESSION = False
RUN_XGBOOST = False
RUN_CNN = True                          # Set to True to run CNN

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# -------------------- CNN Model Definition --------------------
class SpectralCNN(nn.Module):
    """1D CNN for spectral data with attention mechanism"""
    def __init__(self, input_length=2041, num_filters=[64, 128, 256], kernel_sizes=[5, 5, 5], dropout=0.3):
        super().__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Conv1d(1, num_filters[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(num_filters[0], num_filters[1], kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2)
        self.bn2 = nn.BatchNorm1d(num_filters[1])
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(num_filters[1], num_filters[2], kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2)
        self.bn3 = nn.BatchNorm1d(num_filters[2])
        self.pool3 = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters[2], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, input_length)
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, input_length)
        
        # Convolutional blocks
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))  # (batch, num_filters[2], 1)
        
        # Flatten
        x = x.squeeze(-1)  # (batch, num_filters[2])
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x.squeeze()

class SpectralCNNWithAttention(nn.Module):
    """1D CNN with self-attention for spectral data"""
    def __init__(self, input_length=2041, num_filters=[64, 128], kernel_size=5, num_heads=4, dropout=0.3):
        super().__init__()
        
        # Convolutional feature extraction
        self.conv1 = nn.Conv1d(1, num_filters[0], kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(num_filters[0], num_filters[1], kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_filters[1])
        self.pool2 = nn.MaxPool1d(2)
        
        # Self-attention over spectral bands
        self.attention = nn.MultiheadAttention(
            embed_dim=num_filters[1], 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(num_filters[1])
        
    def forward(self, x):
        # x: (batch, input_length)
        x = x.unsqueeze(1)  # (batch, 1, input_length)
        
        # Convolutional feature extraction
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))  # (batch, num_filters[1], seq_len)
        
        # Prepare for attention: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, seq_len, num_filters[1])
        
        # Self-attention
        x, attention_weights = self.attention(x, x, x)
        x = self.layer_norm(x)  # Residual-like connection (simplified)
        
        # Global pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, num_filters[1])
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x.squeeze(), attention_weights

# -------------------- Load Data with proper decimal handling --------------------
print("Loading data...")
# First, read only the first few rows to get column names
col_preview = pd.read_csv(DATA_FILE, sep=';', nrows=0)
all_columns = col_preview.columns.tolist()
print(f"Total columns: {len(all_columns)}")

# Now read the full data, with comma as decimal separator
df = pd.read_csv(DATA_FILE, sep=';', decimal=',', low_memory=False)

# Identify spectral columns (all columns whose name can be converted to int and are within range)
spectral_cols = []
for col in df.columns:
    try:
        val = int(col)
        if SPECTRAL_START <= val <= SPECTRAL_END:
            spectral_cols.append(col)
    except ValueError:
        pass
print(f"Found {len(spectral_cols)} spectral bands.")

if TARGET_COL not in df.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found. Available columns: {df.columns[:20].tolist()}...")

# Drop rows with missing TOC
initial_rows = len(df)
df = df.dropna(subset=[TARGET_COL]).copy()
print(f"Dropped {initial_rows - len(df)} rows with missing TOC. Remaining: {len(df)}")

# Convert spectral columns to float32
X = df[spectral_cols].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"TOC range: {y.min():.3f} - {y.max():.3f}")

# -------------------- Scale features --------------------
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# -------------------- Cross-Validation --------------------
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

results = {}

# --- Linear Regression (if enabled) ---
if RUN_LINEAR_REGRESSION:
    print("\n--- Linear Regression (5‑fold CV) ---")
    lr_scores = []
    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_val)
        # Back‑transform to original scale
        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_val_inv = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
        r2 = r2_score(y_val_inv, y_pred_inv)
        lr_scores.append(r2)
    results['Linear Regression'] = {
        'mean': np.mean(lr_scores),
        'std': np.std(lr_scores),
        'scores': lr_scores
    }
    print(f"Linear Regression R²: {results['Linear Regression']['mean']:.4f} ± {results['Linear Regression']['std']:.4f}")

# --- XGBoost (if enabled) ---
if RUN_XGBOOST:
    print("\n--- XGBoost (5‑fold CV) ---")
    xgb_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED
    }
    xgb_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        xgb_scores.append(r2)
        print(f"  Fold {fold+1} R²: {r2:.4f}")
    results['XGBoost'] = {
        'mean': np.mean(xgb_scores),
        'std': np.std(xgb_scores),
        'scores': xgb_scores
    }
    print(f"XGBoost mean R²: {results['XGBoost']['mean']:.4f} ± {results['XGBoost']['std']:.4f}")

# --- CNN (if enabled) ---
if RUN_CNN:
    print("\n--- 1D CNN (5‑fold CV) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    cnn_scores = []
    cnn_times = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        print(f"\nFold {fold+1}/{N_FOLDS}")
        fold_start = time.time()
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model (choose architecture)
        model = SpectralCNNWithAttention(input_length=X.shape[1]).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0
        best_state = None
        
        for epoch in range(200):
            # Training
            model.train()
            train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred, _ = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred, _ = model(xb)
                    loss = criterion(pred, yb)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if (epoch+1) % 20 == 0:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch+1) % 20 == 0:
                print(f"  Epoch {epoch+1}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")
        
        # Load best model
        model.load_state_dict(best_state)
        
        # Evaluate on validation set
        model.eval()
        all_preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                pred, _ = model(xb)
                all_preds.append(pred.cpu().numpy())
        
        y_pred_scaled = np.concatenate(all_preds)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
        
        r2 = r2_score(y_val_orig, y_pred)
        cnn_scores.append(r2)
        
        fold_time = time.time() - fold_start
        cnn_times.append(fold_time)
        
        print(f"  Fold {fold+1} R²: {r2:.4f} (time: {fold_time:.1f}s)")
    
    results['CNN'] = {
        'mean': np.mean(cnn_scores),
        'std': np.std(cnn_scores),
        'scores': cnn_scores,
        'times': cnn_times
    }
    print(f"\nCNN mean R²: {results['CNN']['mean']:.4f} ± {results['CNN']['std']:.4f}")
    print(f"Average time per fold: {np.mean(cnn_times):.1f}s")

# -------------------- Model Comparison Plot --------------------
if len(results) > 1:
    print("\n--- Generating model comparison plot ---")
    plt.figure(figsize=(10, 6))
    
    models = list(results.keys())
    means = [results[m]['mean'] for m in models]
    stds = [results[m]['std'] for m in models]
    
    bars = plt.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
    plt.ylabel('R² Score')
    plt.title('Model Comparison for TOC Prediction')
    plt.ylim([min(0, min(means) - 0.1), max(means) + 0.1])
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'))
    print(f"Model comparison plot saved to {RESULTS_DIR}/model_comparison.png")

# -------------------- Train Final CNN Model on All Data --------------------
if RUN_CNN:
    print("\n--- Training final CNN on all data ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use the same architecture as during CV
    final_model = SpectralCNNWithAttention(input_length=X.shape[1]).to(device)
    
    # Convert all data to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Training
    final_model.train()
    for epoch in range(100):  # you can increase if desired
        train_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = final_model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss = {train_loss/len(loader):.4f}")
    
    # Save model
    torch.save(final_model.state_dict(), os.path.join(MODEL_DIR, 'soilwise_cnn.pth'))
    print(f"CNN model saved to {MODEL_DIR}/soilwise_cnn.pth")

# -------------------- Summary Report --------------------
print("\n" + "="*50)
print("PROJECT SUMMARY")
print("="*50)
print(f"Dataset: SoilWise EPO-processed spectra")
print(f"Samples: {len(X)}")
print(f"Spectral bands: {len(spectral_cols)}")
print(f"Target: TOC (range: {y.min():.3f} - {y.max():.3f})")
print("\nModel Performance:")
for model_name, metrics in results.items():
    print(f"  {model_name}: R² = {metrics['mean']:.4f} ± {metrics['std']:.4f}")
print("="*50)

# Save results to file
results_summary = pd.DataFrame({
    'Model': list(results.keys()),
    'Mean_R2': [results[m]['mean'] for m in results.keys()],
    'Std_R2': [results[m]['std'] for m in results.keys()]
})
results_summary.to_csv(os.path.join(RESULTS_DIR, 'model_results.csv'), index=False)
print(f"\nResults saved to {RESULTS_DIR}/model_results.csv")