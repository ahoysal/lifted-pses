import copy

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
        graphLevel = (not hasattr(data, "train_mask")) or (data.train_mask is None)

        x = super().forward(data.x, data.edge_index, batch=data.batch if graphLevel else None)

        if graphLevel:
            x = global_add_pool(x, data.batch)
        
        return x

class EdgeBiasedTransformerEncoder(nn.Module):
    """
    Transformer encoder stack that accepts an explicit additive attention bias
    (attn_bias) in addition to the standard padding mask. When attn_bias is None
    the behaviour is identical to nn.TransformerEncoder, so this class is a strict
    superset that is safe to use as a drop-in replacement.

    Bypasses PyTorch's fast-path to guarantee the additive bias is applied correctly
    regardless of PyTorch version.
    """

    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            # Call sublayers directly instead of layer(...) to bypass PyTorch's
            # torch._transformer_encoder_layer_fwd C++ fast path, which is triggered
            # in eval+no_grad mode and mishandles a [B*nhead, N, N] float attn_mask,
            # silently producing NaN.
            x = layer.norm1(x + layer._sa_block(x, attn_bias, src_key_padding_mask))
            x = layer.norm2(x + layer._ff_block(x))
        return x


class GraphNodeTransformer(nn.Module):
    def __init__(self, in_dim, d_model, nhead, num_layers, out_dim=None, dropout=0.1, bond_dim=0):
        """
        Args:
            in_dim:     Input feature size (F).
            d_model:    Internal transformer dimension.
            nhead:      Number of attention heads.
            num_layers: Number of transformer encoder layers.
            out_dim:    Output dimension (defaults to d_model if None).
            bond_dim:   Number of distinct bond/edge types (0 = disabled, no edge features).
                        When > 0 an EdgeBiasedTransformerEncoder is used and a learned
                        additive attention bias is constructed from data.edge_attr at
                        runtime.  Index 0 of the embedding table is the "no-bond" bias
                        applied to non-adjacent pairs; indices 1..bond_dim correspond to
                        actual bond types (1-indexed, as in ZINC).
        """
        super().__init__()
        self.nhead = nhead
        self.bond_dim = bond_dim

        # 1. Project input features to transformer dimension
        self.input_proj = nn.Linear(in_dim, d_model)

        # 2. Transformer encoder — edge-biased when bond_dim > 0
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = EdgeBiasedTransformerEncoder(encoder_layer, num_layers)

        # 3. Output projection
        self.output_proj = nn.Linear(d_model, out_dim if out_dim else d_model)

        # 4. Edge-type embedding table (only allocated when bond features are used)
        #    Shape: [bond_dim + 1, nhead] — one scalar bias per head per edge type.
        #    Index 0: learned "no-bond" bias for all non-adjacent pairs.
        #    Indices 1..bond_dim: learned bias for each bond type.
        if bond_dim > 0:
            self.edge_embed = nn.Embedding(bond_dim + 1, nhead)

    def _build_edge_bias(
        self, data, B: int, N: int, device: torch.device
    ) -> torch.Tensor:
        """
        Construct a dense [B*nhead, N, N] additive attention bias from the sparse
        edge_index / edge_attr stored on data.

        Every pair (i, j) starts with the learned no-bond bias (embedding index 0).
        Pairs that correspond to actual edges are overwritten with the embedding for
        their bond type (embedding indices 1..bond_dim).

        Args:
            data:   Batched PyG Data object.
            B:      Number of graphs in the batch.
            N:      Padded sequence length (max nodes across graphs in batch).
            device: Target device.

        Returns:
            Tensor of shape [B * nhead, N, N].
        """
        nhead = self.nhead

        # Initialise all pairs to the no-bond embedding (index 0): [B, N, N, nhead]
        no_bond_bias = self.edge_embed.weight[0]                        # [nhead]
        bias = no_bond_bias.view(1, 1, 1, nhead).expand(B, N, N, nhead).clone()

        edge_index = data.edge_index                                    # [2, E]
        edge_attr  = data.edge_attr                                     # [E]

        # Map global node indices -> (batch_item, local_node_idx)
        # data.ptr[i] is the first global node index of graph i.
        src_global = edge_index[0]
        dst_global = edge_index[1]
        b_idx      = data.batch[src_global]                             # [E]
        src_local  = src_global - data.ptr[b_idx]                      # [E]
        dst_local  = dst_global - data.ptr[b_idx]                      # [E]

        # Clamp to valid embedding range (ZINC uses 1-indexed bond types)
        bond_types = edge_attr.long().clamp(1, self.bond_dim)           # [E]
        bond_emb   = self.edge_embed(bond_types)                        # [E, nhead]

        bias[b_idx, src_local, dst_local] = bond_emb

        # [B, N, N, nhead] -> [B, nhead, N, N] -> [B*nhead, N, N]
        return bias.permute(0, 3, 1, 2).reshape(B * nhead, N, N)

    def forward(self, data):
        """
        Args:
            data:
                data.x:         [num_nodes, F] node features.
                data.edge_index: [2, E] (required when bond_dim > 0).
                data.edge_attr:  [E] integer bond types (required when bond_dim > 0).
                data.batch:      [num_nodes] batch assignment vector.
                data.ptr:        [B+1] cumulative node counts (present in batched data).
        Returns:
            Tensor of shape [B, out_dim] for graph-level tasks.
        """
        graphLevel = (not hasattr(data, "train_mask")) or (data.train_mask is None)

        x = self.input_proj(data.x)

        attn_bias = None
        if graphLevel:
            x, mask = to_dense_batch(x, data.batch)    # [B, N, d_model]
            B, N = x.shape[:2]

            if (
                self.bond_dim > 0
                and hasattr(data, "edge_attr")
                and data.edge_attr is not None
                and hasattr(data, "ptr")
            ):
                attn_bias = self._build_edge_bias(data, B, N, x.device)
                # Bake padding into the float bias so both masks have the same type.
                # Padding positions become -inf so they are zeroed out by softmax.
                pad = (~mask).unsqueeze(1).unsqueeze(2).expand(B, self.nhead, N, N)
                attn_bias = attn_bias.masked_fill(
                    pad.reshape(B * self.nhead, N, N), float("-inf")
                )
                x = self.transformer(x, attn_bias=attn_bias, src_key_padding_mask=None)
            else:
                # Original path: bool padding mask, no attn_bias — identical to pre-change.
                x = self.transformer(x, attn_bias=None, src_key_padding_mask=~mask)

            x = x[mask]
        else:
            x = self.transformer(x)

        x = self.output_proj(x)

        if graphLevel:
            x = global_add_pool(x, data.batch)

        return x
    

class MeanGuesser(nn.Module):
    def __init__(self, prediction):
        """
        Args:
            prediction: The prediction to return for all
        """
        super().__init__()
        if not isinstance(prediction, torch.Tensor):
            prediction = torch.tensor([prediction], dtype=torch.float32)

        self.pred = nn.Parameter(prediction)

    def forward(self, data):
        """
        Args:
            data: 
                data.x: Tensor of shape [N, F] representing N nodes with F features.
        Returns:
            Tensor of shape [N, 1]
        """
        graphLevel = (not hasattr(data, "train_mask")) or (data.train_mask is None)

        toReturn = 1
        if graphLevel:
            canidate = getattr(data, 'num_graphs', 1)
            if canidate is not None:
                toReturn = canidate
        else:
            canidate = getattr(data, 'num_nodes', 1)
            if canidate is not None:
                toReturn = canidate
            else:
                toReturn = data.x.shape[0]
        
        return self.pred.view(1, -1).expand(toReturn, -1)

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