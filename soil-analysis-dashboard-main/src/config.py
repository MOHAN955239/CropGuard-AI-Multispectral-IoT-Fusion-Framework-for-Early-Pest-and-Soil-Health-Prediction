import torch

# Paths
DATA_PATH = "data/final_sk4.csv"
MODEL_SAVE_PATH = "models/best_model.pth"

# Data parameters
SEQUENCE_LENGTH = 24          # use past 24 timesteps to predict next
TARGET_COL = "SM_20cm"        # soil moisture at 20 cm
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 64

# Model hyperparameters
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.2

# Training
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42