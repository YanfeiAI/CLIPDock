import torch
import torch.nn as nn
import torch.nn.functional as F

from clipdock.atom_type import EMBED_ATOM_TYPE_NUM, EMBED_GAUSS_NUM


class MlpBlock(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 2 * hidden_dim)
        self.layer_norm = nn.LayerNorm(2 * hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.layer_norm(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.dropout(x)
        x = self.fc2(x)
        return x + self.proj(residual)


class GScoreModel(nn.Module):
    def __init__(self, atom_type_num=EMBED_ATOM_TYPE_NUM, gauss_num=EMBED_GAUSS_NUM, out_dim=1,
                 hidden_dim=256, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.atom_type_num = atom_type_num
        self.gauss_num = gauss_num
        self.radial_dim = atom_type_num * self.gauss_num

        self.atom_type_embedding = nn.Embedding(atom_type_num, hidden_dim)
        self.block = nn.Sequential(
            MlpBlock(hidden_dim, self.radial_dim, 4 * hidden_dim, dropout)
        )

    def calc_w(self, atom_idx):
        atom_emb = self.atom_type_embedding(atom_idx)
        w = self.block(atom_emb) / 100
        return w

    def decode(self):
        atom_idx = torch.arange(self.atom_type_num, device=self.atom_type_embedding.weight.device)
        w = self.calc_w(atom_idx)
        w = w.reshape(self.atom_type_num, self.atom_type_num, -1)
        return w

    def forward(self, x):
        atom_idx = torch.arange(self.atom_type_num, device=self.atom_type_embedding.weight.device)
        w = self.calc_w(atom_idx)
        outputs = torch.sum(x * w, dim=-1)
        outputs = torch.sum(outputs, dim=-1).unsqueeze(-1)
        return outputs
