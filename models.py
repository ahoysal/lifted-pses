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

from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.models import GCN as BaseGCN
from torch_geometric.utils import to_dense_batch

class GCN(BaseGCN):
    def __init__(self, *args, **kwargs):
        """
        Args:
            in_dim: Input feature size (F)
            d_model: Internal transformer dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            out_dim: Output dimension (defaults to d_model if None)
        """
        super().__init__(*args, **kwargs)

    def forward(self, data):
        """
        Args:
            data: 
                data.x: Tensor of shape [N, F] representing N nodes with F features.
                (optional) data.batch: if None or doesn't exist, no pooling. else pool
        Returns:
            Tensor of shape [N, out_dim]
        """
        batching = hasattr(data, "batch") and data.batch is not None

        x = super().forward(data.x, data.edge_index, batch=data.batch if batching else None)

        if batching:
            x = global_add_pool(x, data.batch)
        
        return x

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

    def forward(self, data):
        """
        Args:
            data: 
                data.x: Tensor of shape [N, F] representing N nodes with F features.
                (optional) data.batch: if None or doesn't exist, no pooling. else pool
        Returns:
            Tensor of shape [N, out_dim]
        """
        # Add fake batch dimension: [N, F] -> [1, N, F]
        # x = x.unsqueeze(0)
        batching = hasattr(data, "batch") and data.batch is not None
        
        # Project and Transform
        x = self.input_proj(data.x)

        x, mask = to_dense_batch(x, data.batch)

        x = self.transformer(x, src_key_padding_mask=~mask)
        x = x[mask]
        x = self.output_proj(x)

        if batching:
            x = global_add_pool(x, data.batch)
        
        # Remove fake batch dimension: [1, N, d_model] -> [N, d_model]
        # x = x.squeeze(0)
        
        return x
    
# currently in a funky state, do not use...
class GPS(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: dict[str, any]):
        super().__init__()

        # self.node_emb = Embedding(28, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        # self.edge_emb = Embedding(4, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn), heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, data):
        """
        NOTE: 
        data must have attributes
            x
            pe
            edge_index
            edge_attr
        and optionally:
            batch
        """

        x_pe = self.pe_norm(data.pe)
        # x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        x = torch.cat((data.x.squeeze(-1), self.pe_lin(x_pe)), 1)
        # edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, data.edge_index, data.batch, edge_attr=data.edge_attr)
        x = global_add_pool(x, data.batch)

        return self.mlp(x)
    

class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1