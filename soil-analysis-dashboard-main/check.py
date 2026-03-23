import torch
import torch.nn as nn
import joblib
import numpy as np

# ---------- Copy the model definition from the app ----------
class FertilityMLP(nn.Module):
    def __init__(self, input_dim=12, hidden_dims=[256, 128, 64], dropout=0.2):
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
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ---------- Paths ----------
MODEL_PATH = "models/mlp_model_improved/fertility_model_final.pth"
SCALER_PATH = "models/mlp_model_improved/scaler.pkl"

# ---------- Load scaler ----------
scaler = joblib.load(SCALER_PATH)
print("Scaler loaded.")
print("Scaler mean_ :", scaler.mean_)
print("Scaler scale_:", scaler.scale_)

# ---------- Load model ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FertilityMLP()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("\nModel loaded successfully.")

# ---------- Sample inputs (as defined in the app) ----------
SAMPLE_LOW = {
    'N': 20.0, 'P': 60.0, 'K': 100.0, 'ph': 7.4, 'ec': 0.4, 'oc': 0.2,
    'S': 8.0, 'zn': 3.0, 'fe': 20.0, 'cu': 0.8, 'Mn': 15.0, 'B': 0.3
}
SAMPLE_MEDIUM = {
    'N': 150.0, 'P': 15.0, 'K': 350.0, 'ph': 7.5, 'ec': 0.6, 'oc': 0.8,
    'S': 10.0, 'zn': 0.5, 'fe': 1.5, 'cu': 0.9, 'Mn': 5.0, 'B': 0.8
}
SAMPLE_HIGH = {
    'N': 300.0, 'P': 10.0, 'K': 500.0, 'ph': 7.6, 'ec': 0.5, 'oc': 1.2,
    'S': 5.0, 'zn': 0.4, 'fe': 0.8, 'cu': 0.7, 'Mn': 3.0, 'B': 1.0
}

FEATURES = ['N', 'P', 'K', 'ph', 'ec', 'oc', 'S', 'zn', 'fe', 'cu', 'Mn', 'B']

def predict(sample_dict):
    arr = np.array([[sample_dict[f] for f in FEATURES]], dtype=np.float32)
    arr_scaled = scaler.transform(arr)
    tensor = torch.tensor(arr_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(tensor).cpu().numpy()[0,0]
    return arr_scaled, pred

print("\n" + "="*50)
for name, sample in [("LOW", SAMPLE_LOW), ("MEDIUM", SAMPLE_MEDIUM), ("HIGH", SAMPLE_HIGH)]:
    scaled, pred = predict(sample)
    print(f"\n{name} sample:")
    print("  Raw inputs :", [sample[f] for f in FEATURES])
    print("  Scaled     :", scaled.flatten())
    print(f"  Prediction : {pred:.6f}")