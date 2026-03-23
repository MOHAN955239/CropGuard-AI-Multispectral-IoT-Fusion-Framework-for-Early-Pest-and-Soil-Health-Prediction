import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.config import *

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path, parse_dates=['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # Interpolate missing values
    df = df.interpolate(method='linear', limit_direction='both')
    
    # Define modalities
    soil_cols = [col for col in df.columns if col.startswith('SM_') or col.startswith('ST_')]
    meteo_cols = ['Air_Temp', 'Precip_Total', 'Wind_Speed', 'Rel_Humidity',
                  'Precip_24h_sum', 'Air_Temp_24h_mean', 'Wind_Speed_24h_mean',
                  'Rel_Humidity_24h_mean', 'day_of_year_sin', 'day_of_year_cos',
                  'hour_sin', 'hour_cos', 'is_frozen', 'Wind_Dir_sin', 'Wind_Dir_cos']
    
    # Remove target from soil_cols if present
    if TARGET_COL in soil_cols:
        soil_cols.remove(TARGET_COL)
    
    # Separate features and target
    X_soil = df[soil_cols].values
    X_meteo = df[meteo_cols].values
    y = df[[TARGET_COL]].values
    
    # Scale each modality separately
    scaler_soil = StandardScaler()
    scaler_meteo = StandardScaler()
    scaler_y = StandardScaler()
    
    X_soil_scaled = scaler_soil.fit_transform(X_soil)
    X_meteo_scaled = scaler_meteo.fit_transform(X_meteo)
    y_scaled = scaler_y.fit_transform(y)
    
    # Create sequences
    def create_sequences(data1, data2, target, seq_len):
        X1, X2, Y = [], [], []
        for i in range(len(data1) - seq_len):
            X1.append(data1[i:i+seq_len])
            X2.append(data2[i:i+seq_len])
            Y.append(target[i+seq_len])   # predict next step
        return np.array(X1), np.array(X2), np.array(Y)
    
    X1, X2, Y = create_sequences(X_soil_scaled, X_meteo_scaled, y_scaled, SEQUENCE_LENGTH)
    
    # Temporal split (no shuffle)
    total = len(X1)
    test_len = int(total * TEST_SIZE)
    val_len = int(total * VAL_SIZE)
    train_len = total - test_len - val_len
    
    X1_train = X1[:train_len]
    X2_train = X2[:train_len]
    Y_train = Y[:train_len]
    
    X1_val = X1[train_len:train_len+val_len]
    X2_val = X2[train_len:train_len+val_len]
    Y_val = Y[train_len:train_len+val_len]
    
    X1_test = X1[train_len+val_len:]
    X2_test = X2[train_len+val_len:]
    Y_test = Y[train_len+val_len:]
    
    # Convert to tensors
    def to_tensor(*arrays):
        return [torch.FloatTensor(arr) for arr in arrays]
    
    X1_train, X2_train, Y_train = to_tensor(X1_train, X2_train, Y_train)
    X1_val, X2_val, Y_val = to_tensor(X1_val, X2_val, Y_val)
    X1_test, X2_test, Y_test = to_tensor(X1_test, X2_test, Y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X1_train, X2_train, Y_train)
    val_dataset = TensorDataset(X1_val, X2_val, Y_val)
    test_dataset = TensorDataset(X1_test, X2_test, Y_test)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Return scalers as well
    return train_loader, val_loader, test_loader, scaler_soil, scaler_meteo, scaler_y, (X1.shape[-1], X2.shape[-1])