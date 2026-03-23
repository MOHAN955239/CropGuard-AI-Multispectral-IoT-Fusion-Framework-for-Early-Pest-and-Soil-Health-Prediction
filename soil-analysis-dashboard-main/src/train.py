import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils import save_model
from src.config import *

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x1, x2, y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        pred, _ = model(x1, x2)
        loss = criterion(pred, y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            pred, _ = model(x1, x2)
            loss = criterion(pred, y.squeeze())
            total_loss += loss.item()
    return total_loss / len(loader)

def train(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, MODEL_SAVE_PATH)
    
    print(f"Training complete. Best val loss: {best_val_loss:.6f}")