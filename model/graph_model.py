import torch
from torch import nn
import torch.nn.functional as F
from model.graph_transformer import GraphTransformer


class CLIPDockVS(nn.Module):
    def __init__(self, prot_node_in, prot_edge_in, lig_node_in, lig_edge_in,
                 hidden_dim=256, num_layers=8, num_gaussians=16, dropout=0.1, dist_cutoff=8.0, num_heads=8):
        super().__init__()
        self.prot_gnn = GraphTransformer(prot_node_in, prot_edge_in, hidden_dim, num_layers, dropout, num_heads)
        self.lig_gnn = GraphTransformer(lig_node_in, lig_edge_in, hidden_dim, num_layers, dropout, num_heads)

        self.num_gaussians = num_gaussians
        self.dist_cutoff = dist_cutoff

        reso = dist_cutoff / num_gaussians
        mu = torch.tensor([reso * (i + 0.5) for i in range(num_gaussians)], dtype=torch.float32)
        sigma = torch.tensor([reso for _ in range(num_gaussians)], dtype=torch.float32)

        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_gaussians)
        )

    def forward(self, prot_x, prot_edge_index, prot_edge_attr, prot_batch,
                lig_x, lig_edge_index, lig_edge_attr, lig_batch):

        h_prot = self.prot_gnn(prot_x, prot_edge_index, prot_edge_attr)
        h_lig = self.lig_gnn(lig_x, lig_edge_index, lig_edge_attr)

        batch_size = int(prot_batch.max().item() + 1)
        device = prot_x.device

        count_p = torch.bincount(prot_batch, minlength=batch_size)
        count_l = torch.bincount(lig_batch, minlength=batch_size)

        repeats_p = count_l[prot_batch]
        idx_p = torch.arange(h_prot.size(0), device=device)
        final_p_idx = idx_p.repeat_interleave(repeats_p)

        l_starts = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        l_starts[1:] = torch.cumsum(count_l, 0)

        total_pairs = final_p_idx.size(0)
        p_res_starts = torch.zeros(h_prot.size(0) + 1, dtype=torch.long, device=device)
        p_res_starts[1:] = torch.cumsum(repeats_p, 0)

        offset_adjustment = p_res_starts[:-1].repeat_interleave(repeats_p)
        local_l_indices = torch.arange(total_pairs, device=device) - offset_adjustment

        sample_of_protein_repeated = prot_batch.repeat_interleave(repeats_p)
        final_l_idx = l_starts[sample_of_protein_repeated] + local_l_indices

        h_pair = torch.cat([h_prot[final_p_idx], h_lig[final_l_idx]], dim=-1)
        pi_logits = self.mlp(h_pair)
        log_pi = F.log_softmax(pi_logits, dim=-1)

        mu = self.mu.expand(log_pi.size(0), -1)
        sigma = self.sigma.expand(log_pi.size(0), -1)

        return log_pi, mu, sigma, (count_p * count_l).tolist()
