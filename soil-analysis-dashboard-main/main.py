from src.config import *
from src.utils import set_seed, load_model
from src.data_loader import load_and_preprocess
from src.model import FusionModel
from src.train import train
from src.evaluate import evaluate

def main():
    set_seed(RANDOM_SEED)
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("Loading and preprocessing data...")
    train_loader, val_loader, test_loader, scaler_y, (input_dim1, input_dim2) = load_and_preprocess(DATA_PATH)
    
    # Initialize model
    model = FusionModel(input_dim1, input_dim2).to(DEVICE)
    print(model)
    
    # Train
    print("\nStarting training...")
    train(model, train_loader, val_loader, DEVICE)
    
    # Load best model and evaluate
    print("\nLoading best model for evaluation...")
    model = load_model(model, MODEL_SAVE_PATH, DEVICE)
    metrics = evaluate(model, test_loader, scaler_y, DEVICE)
    
    print("\nProject completed successfully!")

if __name__ == "__main__":
    main()