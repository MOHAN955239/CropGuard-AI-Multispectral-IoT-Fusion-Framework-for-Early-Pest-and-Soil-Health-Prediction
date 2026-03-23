# Soil Analysis Dashboard

This project combines two soil prediction models into a single Streamlit dashboard:

1. **Multi‑modal soil moisture prediction** (GRU + attention) using sensor data.
2. **Hyperspectral CNN** for Total Organic Carbon (TOC) prediction from soil spectra.

## Features
- Predict soil moisture at 20 cm depth from time‑series sensor data.
- Predict TOC from a single soil spectrum (2041 bands, 405–2445 nm).
- Switch between models via a sidebar.

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt

Usage
Run the dashboard:

bash
streamlit run app_combined.py
Models
Multi‑modal model: models/best_model.pth and associated scalers.

CNN model: models/cnn/soilwise_cnn.pth and scalers.

Data
The multi‑modal model expects a CSV with columns as in final_sk4.csv.

The CNN expects a single‑row CSV with 2041 reflectance values.

Results
Multi‑modal model achieved R² ≈ 0.91 on test set.

CNN achieved R² ≈ 0.70 (can be improved with more data).