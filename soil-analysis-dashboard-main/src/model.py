import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import *

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key, value):
        attn_output, attn_weights = self.mha(query, key, value)
        return self.layer_norm(query + attn_output), attn_weights

class FusionModel(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim=HIDDEN_DIM, 
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT):
        super().__init__()
        # Encoders
        self.encoder1 = nn.GRU(input_dim1, hidden_dim, num_layers, 
                               batch_first=True, dropout=dropout)
        self.encoder2 = nn.GRU(input_dim2, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout)
        
        # Cross-attention
        self.cross_att1 = MultiHeadCrossAttention(hidden_dim, num_heads)
        self.cross_att2 = MultiHeadCrossAttention(hidden_dim, num_heads)
        
        # Fusion & regression
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x1, x2):
        # x1: (batch, seq_len, input_dim1), x2: (batch, seq_len, input_dim2)
        out1, _ = self.encoder1(x1)   # (batch, seq_len, hidden_dim)
        out2, _ = self.encoder2(x2)
        
        # Cross-attention: each modality attends to the other
        attended1, attn1 = self.cross_att1(out1, out2, out2)  # query from out1
        attended2, attn2 = self.cross_att2(out2, out1, out1)  # query from out2
        
        # Use last time step
        pooled1 = attended1[:, -1, :]
        pooled2 = attended2[:, -1, :]
        
        # Concatenate and predict
        combined = torch.cat([pooled1, pooled2], dim=-1)
        output = self.fc(combined)
        return output.squeeze(), (attn1, attn2)  # return attention for analysis