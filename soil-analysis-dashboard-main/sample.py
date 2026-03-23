import pandas as pd
import numpy as np
import os

# Path to your original dataset
data_path = r"D:\Crop Final\data\D3.2_20240117_ProbeField_Preprosessed_Spectra_DT_EPO_V1.csv"

# Load the full dataset (this may take a moment)
print("Loading dataset...")
df = pd.read_csv(data_path, sep=';', decimal=',', low_memory=False)

# Identify spectral columns (wavelengths 405–2445)
spectral_cols = [c for c in df.columns if c.isdigit() and 405 <= int(c) <= 2445]
print(f"Found {len(spectral_cols)} spectral bands.")

# Drop rows where TOC is missing
df = df.dropna(subset=['TOC']).reset_index(drop=True)

# Convert TOC to numeric (handles comma decimals)
df['TOC'] = pd.to_numeric(df['TOC'], errors='coerce')

# Number of samples to extract
n_samples = 5
# Randomly select indices
np.random.seed(42)  # for reproducibility
indices = np.random.choice(len(df), size=n_samples, replace=False)

# Create output directory if needed
output_dir = "cnn_samples"
os.makedirs(output_dir, exist_ok=True)

print("\nExtracting samples:")
for i, idx in enumerate(indices, 1):
    row = df.iloc[idx]
    spectrum = row[spectral_cols].values.astype(float)
    toc = row['TOC']
    
    # Save to CSV (one row, no header)
    filename = os.path.join(output_dir, f"sample_{i}.csv")
    with open(filename, 'w') as f:
        f.write(','.join(f'{x:.6f}' for x in spectrum))
    
    print(f"  Sample {i}: index {idx}, TOC = {toc:.3f}%, saved to {filename}")

print(f"\n✅ {n_samples} sample files created in the '{output_dir}' folder.")