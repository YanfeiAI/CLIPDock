import torch
from torch import nn
import torch.nn.functional as F

class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.O_node = nn.Linear(hidden_dim, hidden_dim)
        self.O_edge = nn.Linear(hidden_dim, hidden_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        )

        self.norm1_node = nn.BatchNorm1d(hidden_dim)
        self.norm1_edge = nn.BatchNorm1d(hidden_dim)
        self.norm2_node = nn.BatchNorm1d(hidden_dim)
        self.norm2_edge = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index

        x_norm = self.norm1_node(x)
        e_norm = self.norm1_edge(edge_attr)

        q = self.Q(x_norm).view(-1, self.num_heads, self.head_dim)
        k = self.K(x_norm).view(-1, self.num_heads, self.head_dim)
        v = self.V(x_norm).view(-1, self.num_heads, self.head_dim)
        proj_e = self.edge_proj(e_norm).view(-1, self.num_heads, self.head_dim)

        k_src = k[row] # [E, H, D]
        q_dst = q[col] # [E, H, D]

        score = k_src * q_dst # [E, H, D]
        score = score / (self.head_dim ** 0.5)
        score = torch.clamp(score, min=-5.0, max=5.0)

        score = score * proj_e # [E, H, D]

        e_out_raw = score.reshape(-1, self.hidden_dim) # [E, hidden_dim]

        attn_logits = score.sum(dim=-1) # [E, H]

        attn_logits_f32 = attn_logits.to(torch.float32)

        num_nodes = x.size(0)
        head_offsets = torch.arange(self.num_heads, device=x.device).view(1, self.num_heads)
        flat_col = col.view(-1, 1) * self.num_heads + head_offsets # [E, H]

        attn_max_init = torch.full((num_nodes * self.num_heads,), -float('inf'), dtype=torch.float32, device=x.device)
        attn_max_flat = attn_max_init.index_reduce(0, flat_col.flatten(), attn_logits_f32.flatten(), 'amax', include_self=False)
        attn_max = attn_max_flat.view(num_nodes, self.num_heads) # [N, H]

        attn_weights = torch.exp(attn_logits_f32 - attn_max[col]) # [E, H]

        dst_sum = torch.zeros(x.size(0), self.num_heads, dtype=torch.float32, device=x.device)
        dst_sum.index_add_(0, col, attn_weights)

        alpha = attn_weights / (dst_sum[col] + 1e-10) # [E, H]
        alpha = alpha.to(score.dtype)

        v_src = v[row] # [E, H, D]
        msg = v_src * alpha.unsqueeze(-1) # [E, H, D]

        h_out = torch.zeros(x.size(0), self.num_heads, self.head_dim, dtype=msg.dtype, device=x.device)
        h_out.index_add_(0, col, msg)
        h_out = h_out.reshape(-1, self.hidden_dim)

        h_out = self.O_node(F.dropout(h_out, p=self.dropout, training=self.training))
        e_out = self.O_edge(F.dropout(e_out_raw, p=self.dropout, training=self.training))

        h = x + h_out
        e = edge_attr + e_out

        h_norm2 = self.norm2_node(h)
        e_norm2 = self.norm2_edge(e)

        h_mlp = self.node_mlp(h_norm2)
        e_mlp = self.edge_mlp(e_norm2)

        h = h + h_mlp
        e = e + e_mlp

        return h, e

class GraphTransformer(nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim, num_layers, dropout=0.1, num_heads=4):
        super().__init__()
        self.node_encoder = nn.Linear(node_in, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in, hidden_dim)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_attr):
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        for layer in self.layers:
            h, e = layer(h, edge_index, e)

        return h
