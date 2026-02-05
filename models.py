import torch
import torch.nn as nn

from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)

from torch_geometric.nn import GINEConv, GPSConv, global_add_pool

class GraphNodeTransformer(nn.Module):
    def __init__(self, in_dim, d_model, nhead, num_layers, out_dim=None, dropout=0.1):
        """
        Args:
            in_dim: Input feature size (F)
            d_model: Internal transformer dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            out_dim: Output dimension (defaults to d_model if None)
        """
        super().__init__()
        
        # 1. Project input features to transformer dimension
        self.input_proj = nn.Linear(in_dim, d_model)
        
        # 2. Standard Transformer Encoder
        # batch_first=True ensures it accepts [Batch, Seq, Feature]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Projection
        self.output_proj = nn.Linear(d_model, out_dim if out_dim else d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [N, F] representing N nodes with F features.
        Returns:
            Tensor of shape [N, out_dim]
        """
        # Add fake batch dimension: [N, F] -> [1, N, F]
        # x = x.unsqueeze(0)
        
        # Project and Transform
        x = self.input_proj(x)
        x = self.transformer(x)
        
        # Remove fake batch dimension: [1, N, d_model] -> [N, d_model]
        # x = x.squeeze(0)
        
        return self.output_proj(x)
    