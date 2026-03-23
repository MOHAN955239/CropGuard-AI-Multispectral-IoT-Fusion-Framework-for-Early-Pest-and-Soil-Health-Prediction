import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.data_loader import load_and_preprocess
from src.model import FusionModel
from src.config import *
from src.utils import load_model

def get_all_predictions(loader, model, device):
    """Run model on entire loader and return predictions + targets."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2 = x1.to(device), x2.to(device)
            pred, _ = model(x1, x2)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_targets)

def main():
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, scaler_soil, scaler_meteo, scaler_y, (input_dim1, input_dim2) = load_and_preprocess(DATA_PATH)
    
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    print(f"Input dimensions - Soil: {input_dim1}, Meteo: {input_dim2}")
    
    # Load model - use the EXACT dimensions from data_loader
    print("Loading model...")
    model = FusionModel(input_dim1=input_dim1, input_dim2=input_dim2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    
    # Get predictions for train and test
    print("Getting predictions...")
    train_preds_scaled, train_targets_scaled = get_all_predictions(train_loader, model, DEVICE)
    test_preds_scaled, test_targets_scaled = get_all_predictions(test_loader, model, DEVICE)
    
    # Inverse transform
    train_preds = scaler_y.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
    train_actual = scaler_y.inverse_transform(train_targets_scaled.reshape(-1, 1)).flatten()
    test_preds = scaler_y.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
    test_actual = scaler_y.inverse_transform(test_targets_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    train_r2 = r2_score(train_actual, train_preds)
    test_r2 = r2_score(test_actual, test_preds)
    train_mae = mean_absolute_error(train_actual, train_preds)
    test_mae = mean_absolute_error(test_actual, test_preds)
    train_rmse = np.sqrt(mean_squared_error(train_actual, train_preds))
    test_rmse = np.sqrt(mean_squared_error(test_actual, test_preds))
    
    # Print results
    print("\n" + "="*60)
    print("MODEL OVERFITTING CHECK")
    print("="*60)
    print(f"{'Metric':<15} {'Train':<15} {'Test':<15} {'Gap':<15}")
    print("-"*60)
    print(f"{'R²':<15} {train_r2:<15.4f} {test_r2:<15.4f} {train_r2 - test_r2:<15.4f}")
    print(f"{'MAE':<15} {train_mae:<15.4f} {test_mae:<15.4f} {train_mae - test_mae:<15.4f}")
    print(f"{'RMSE':<15} {train_rmse:<15.4f} {test_rmse:<15.4f} {train_rmse - test_rmse:<15.4f}")
    print("="*60)
    
    # Interpretation
    if train_r2 - test_r2 > 0.05:
        print("⚠️  SIGNIFICANT OVERFITTING: Train-test R² gap > 0.05")
    elif train_r2 - test_r2 > 0.02:
        print("⚠️  MILD OVERFITTING: Train-test R² gap between 0.02-0.05")
    else:
        print("✅ NO SIGNIFICANT OVERFITTING: Train-test R² gap < 0.02")
    
    if test_mae > 1.5 * train_mae:
        print(f"⚠️  Test MAE is {test_mae/train_mae:.1f}x train MAE")

if __name__ == "__main__":
    main()