import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# -------------------- Configuration --------------------
DATA_FILE = r"D:\Crop Final\data\Soil Fertility Data (Modified Data).csv"  # <-- UPDATE PATH
TARGET_COL = "fertility"
FEATURE_COLS = ['N', 'P', 'K', 'ph', 'ec', 'oc', 'S', 'zn', 'fe', 'cu', 'Mn', 'B']
TEST_SIZE = 0.15
RANDOM_SEED = 42
MODEL_SAVE_DIR = "models/mlp_model_improved"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# -------------------- Optional: Clip outliers (using percentiles) --------------------
CLIP_OUTLIERS = True          # set to False if you don't want clipping
LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99

# -------------------- Load and prepare data --------------------
print("Loading data...")
df = pd.read_csv(DATA_FILE)

# Check required columns
for col in FEATURE_COLS + [TARGET_COL]:
    if col not in df.columns:
        raise KeyError(f"Column {col} not found in data.")

# Drop rows with missing values
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

# Map fertility classes to continuous scores (0→0, 1→0.5, 2→1)
df['fertility_score'] = df[TARGET_COL].map({0: 0.0, 1: 0.5, 2: 1.0})

X = df[FEATURE_COLS].values.astype(np.float32)
y = df['fertility_score'].values.astype(np.float32)

print(f"Total samples: {len(X)}")

# Optional: clip extreme values to reduce outlier influence
if CLIP_OUTLIERS:
    for i, col in enumerate(FEATURE_COLS):
        lower = np.percentile(X[:, i], LOWER_PERCENTILE)
        upper = np.percentile(X[:, i], UPPER_PERCENTILE)
        X[:, i] = np.clip(X[:, i], lower, upper)
    print("Outliers clipped to percentiles [{}, {}]".format(LOWER_PERCENTILE, UPPER_PERCENTILE))

# -------------------- Train / Test split --------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# Scale features (fit on training portion only)
scaler = StandardScaler()
X_temp_scaled = scaler.fit_transform(X_temp)
X_test_scaled = scaler.transform(X_test)

# Save scaler for later use
joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler.pkl"))

# -------------------- Define PyTorch model (flexible) --------------------
class FertilityMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
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
        layers.append(nn.Sigmoid())          # outputs in [0,1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------- Hyperparameter tuning (grid search with cross-validation) --------------------
print("\nStarting hyperparameter tuning...")

# Define a small grid of hyperparameters to search
param_grid = {
    'hidden_layers': [[64, 32], [128, 64, 32], [256, 128, 64]],
    'dropout': [0.2, 0.3, 0.4],
    'lr': [1e-3, 5e-4],
    'batch_size': [32, 64]
}

# We'll use 3-fold CV on the training portion (X_temp_scaled, y_temp)
kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

best_score = -np.inf
best_params = None

# Fixed training parameters for CV folds
EPOCHS_PER_FOLD = 100
PATIENCE = 10

# Prepare data as tensors once (will be re-split in each fold)
X_temp_tensor = torch.tensor(X_temp_scaled, dtype=torch.float32)
y_temp_tensor = torch.tensor(y_temp, dtype=torch.float32).unsqueeze(1)

for hidden in param_grid['hidden_layers']:
    for dropout in param_grid['dropout']:
        for lr in param_grid['lr']:
            for batch_size in param_grid['batch_size']:
                fold_scores = []
                print(f"\nTesting: hidden={hidden}, dropout={dropout}, lr={lr}, batch_size={batch_size}")
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp_tensor)):
                    # Split data
                    X_tr = X_temp_tensor[train_idx]
                    y_tr = y_temp_tensor[train_idx]
                    X_val = X_temp_tensor[val_idx]
                    y_val = y_temp_tensor[val_idx]

                    # Create data loaders
                    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                                              batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(TensorDataset(X_val, y_val),
                                            batch_size=batch_size, shuffle=False)

                    # Initialize model
                    model = FertilityMLP(input_dim=len(FEATURE_COLS),
                                         hidden_dims=hidden,
                                         dropout=dropout).to(DEVICE)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                     factor=0.5, patience=5)

                    best_val_loss = float('inf')
                    patience_counter = 0
                    for epoch in range(EPOCHS_PER_FOLD):
                        # Training
                        model.train()
                        train_loss = 0
                        for xb, yb in train_loader:
                            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                            optimizer.zero_grad()
                            pred = model(xb)
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
                                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                                pred = model(xb)
                                loss = criterion(pred, yb)
                                val_loss += loss.item()
                        val_loss /= len(val_loader)

                        scheduler.step(val_loss)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= PATIENCE:
                                break

                    # After training, compute R² on validation set
                    model.eval()
                    with torch.no_grad():
                        preds = model(X_val.to(DEVICE)).cpu().numpy()
                    r2 = r2_score(y_val.numpy(), preds)
                    fold_scores.append(r2)
                    print(f"  Fold {fold+1}: R² = {r2:.4f}")

                avg_r2 = np.mean(fold_scores)
                print(f"  Average R² = {avg_r2:.4f}")
                if avg_r2 > best_score:
                    best_score = avg_r2
                    best_params = {
                        'hidden_layers': hidden,
                        'dropout': dropout,
                        'lr': lr,
                        'batch_size': batch_size
                    }

print(f"\nBest hyperparameters: {best_params}")
print(f"Best CV R² = {best_score:.4f}")

# -------------------- Retrain final model on full training set --------------------
print("\nTraining final model on full training set...")
final_model = FertilityMLP(input_dim=len(FEATURE_COLS),
                           hidden_dims=best_params['hidden_layers'],
                           dropout=best_params['dropout']).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

train_dataset = TensorDataset(X_temp_tensor, y_temp_tensor)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
val_dataset = TensorDataset(X_temp_tensor, y_temp_tensor)  # we use same for monitoring (no separate val)
val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)

EPOCHS_FINAL = 300
PATIENCE_FINAL = 30
best_val_loss = float('inf')
patience_counter = 0
best_model_path = os.path.join(MODEL_SAVE_DIR, "fertility_model_best.pth")

train_losses, val_losses = [], []

for epoch in range(EPOCHS_FINAL):
    # Training
    final_model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = final_model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation (on the same training set – just for monitoring, no early stopping on it)
    final_model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = final_model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    # Use validation loss for early stopping (though it's the same set, still useful)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(final_model.state_dict(), best_model_path)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE_FINAL:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

# Load best model
final_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
final_model.eval()

# -------------------- Evaluate on test set --------------------
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    preds_test = final_model(X_test_tensor.to(DEVICE)).cpu().numpy().flatten()
    targets_test = y_test_tensor.numpy().flatten()

r2 = r2_score(targets_test, preds_test)
mae = mean_absolute_error(targets_test, preds_test)
rmse = np.sqrt(mean_squared_error(targets_test, preds_test))

print("\n" + "="*40)
print("TEST SET PERFORMANCE")
print("="*40)
print(f"R²  = {r2:.4f}")
print(f"MAE = {mae:.4f} (score units)")
print(f"RMSE= {rmse:.4f}")
print("="*40)

# -------------------- Feature Importance (Manual Permutation) --------------------
print("\nComputing permutation feature importance on test set manually...")

n_repeats = 10
baseline_r2 = r2
print(f"Baseline R² on test set: {baseline_r2:.4f}\n")

importance_mean = []
importance_std = []

X_test_np = X_test_scaled.copy()
rng = np.random.RandomState(RANDOM_SEED)

for i, col in enumerate(FEATURE_COLS):
    scores = []
    for r in range(n_repeats):
        X_permuted = X_test_np.copy()
        rng.shuffle(X_permuted[:, i])                # shuffle one feature
        with torch.no_grad():
            preds_perm = final_model(torch.tensor(X_permuted, dtype=torch.float32).to(DEVICE)).cpu().numpy().flatten()
        score = r2_score(targets_test, preds_perm)
        scores.append(score)
    mean_drop = baseline_r2 - np.mean(scores)
    std_drop = np.std(scores)
    importance_mean.append(mean_drop)
    importance_std.append(std_drop)
    print(f"{col:>4}: {mean_drop:.4f} ± {std_drop:.4f}")

# Convert to numpy arrays for easy indexing
importance_mean = np.array(importance_mean)
importance_std = np.array(importance_std)

# Sort features by importance
sorted_idx = np.argsort(importance_mean)[::-1]
print("\nSorted feature importance (mean R² drop):")
for idx in sorted_idx:
    print(f"{FEATURE_COLS[idx]:>4}: {importance_mean[idx]:.4f} ± {importance_std[idx]:.4f}")

# Plot
plt.figure(figsize=(10,6))
plt.barh(np.array(FEATURE_COLS)[sorted_idx], importance_mean[sorted_idx],
         xerr=importance_std[sorted_idx])
plt.xlabel('Mean R² decrease (higher → more important)')
plt.title('Permutation Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_DIR, 'feature_importance.png'))
plt.show()

# -------------------- Sample Inputs for Low/Medium/High Fertility --------------------
print("\nGenerating sample inputs based on dataset percentiles...")

# Get percentiles from original (unscaled) training data (X_temp)
low_sample = {}
med_sample = {}
high_sample = {}

for i, col in enumerate(FEATURE_COLS):
    low_sample[col] = np.percentile(X_temp[:, i], 10)   # low fertility
    med_sample[col] = np.percentile(X_temp[:, i], 50)   # medium fertility
    high_sample[col] = np.percentile(X_temp[:, i], 90)  # high fertility

# Override some columns for high sample based on typical high-fertility observations:
high_sample['N'] = 300
high_sample['P'] = 10      # low P is typical for high fertility in this dataset
high_sample['K'] = 500
high_sample['zn'] = 0.5
high_sample['fe'] = 1.0
high_sample['Mn'] = 5.0

# For low sample, use extreme low/high values
low_sample['N'] = 20
low_sample['P'] = 60       # high P associated with low fertility
low_sample['K'] = 100
low_sample['oc'] = 0.2
low_sample['zn'] = 3.0
low_sample['fe'] = 20.0
low_sample['Mn'] = 15.0

# Function to predict for a single sample
def predict_fertility(feature_dict):
    """Input: dict with keys = FEATURE_COLS. Returns predicted score."""
    # Convert to array in correct order
    arr = np.array([[feature_dict[col] for col in FEATURE_COLS]], dtype=np.float32)
    # Scale using fitted scaler
    arr_scaled = scaler.transform(arr)
    # Predict
    with torch.no_grad():
        pred = final_model(torch.tensor(arr_scaled, dtype=torch.float32).to(DEVICE)).cpu().numpy()[0,0]
    return pred

# Predict for each sample
print("\nSample input – Low fertility:")
for col in FEATURE_COLS:
    print(f"  {col}: {low_sample[col]:.2f}")
print(f"Predicted fertility score: {predict_fertility(low_sample):.3f}")

print("\nSample input – Medium fertility:")
for col in FEATURE_COLS:
    print(f"  {col}: {med_sample[col]:.2f}")
print(f"Predicted fertility score: {predict_fertility(med_sample):.3f}")

print("\nSample input – High fertility:")
for col in FEATURE_COLS:
    print(f"  {col}: {high_sample[col]:.2f}")
print(f"Predicted fertility score: {predict_fertility(high_sample):.3f}")

# -------------------- Save final model and related objects --------------------
torch.save(final_model.state_dict(), os.path.join(MODEL_SAVE_DIR, "fertility_model_final.pth"))
joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler.pkl"))
print(f"\nFinal model and scaler saved to {MODEL_SAVE_DIR}")