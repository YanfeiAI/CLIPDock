import torch
import numpy as np
import math
import logging
from rdkit import RDLogger

def disable_rdkit_logging():
    rdkit_logger = logging.getLogger('rdkit')
    rdkit_logger.disabled = True
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')

def to_numpy(d):
    if isinstance(d, dict):
        return {k: to_numpy(v) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d.numpy()
    elif isinstance(d, list):
        return [to_numpy(i) for i in d]
    return d

def to_tensor(d):
    if isinstance(d, dict):
        return {k: to_tensor(v) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        if d.dtype == np.float64:
            d = d.astype(np.float32)
        if d.dtype == np.int64 and d.ndim > 0:
            return torch.from_numpy(d)
        return torch.from_numpy(d)
    elif isinstance(d, list):
        return [to_tensor(i) for i in d]
    return d

def collate_graph_attributes(batch, key):
    batch = [to_tensor(g) for g in batch]

    num_nodes = [g[key]['num_nodes'] for g in batch]
    cum_nodes = torch.cumsum(torch.tensor([0] + num_nodes), dim=0)

    x = torch.cat([g[key]['x'] for g in batch], dim=0)

    edge_indices = [g[key]['edge_index'] + cum_nodes[i] for i, g in enumerate(batch)]
    edge_index = torch.cat(edge_indices, dim=1)

    edge_attr = torch.cat([g[key]['edge_attr'] for g in batch], dim=0)

    batch_idx = torch.cat([torch.full((num_nodes[i],), i, dtype=torch.long) for i in range(len(batch))], dim=0)

    pos = torch.cat([g[key]['pos'] for g in batch], dim=0)

    result = {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'batch': batch_idx, 'pos': pos}

    return result

def get_cosine_schedule_with_warmup(optimizer, num_training_steps, warm_up_ratio=0.1, end_lr_ratio=0.1):
    def lr_lambda(current_step):
        if num_training_steps is None or num_training_steps == 0:
            return 1.0

        progress = current_step / num_training_steps
        if progress < warm_up_ratio:
            return progress / warm_up_ratio
        else:
            cosine_progress = (progress - warm_up_ratio) / (1 - warm_up_ratio)
            return end_lr_ratio + (1 - end_lr_ratio) * 0.5 * (1.0 + math.cos(cosine_progress * math.pi))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def mdn_loss_fn(log_pi, mu, sigma, target_dist, dist_threshold=8.0):
    log_pi = log_pi.to(torch.float32)
    mu = mu.to(torch.float32)
    sigma = sigma.to(torch.float32)
    target_dist = target_dist.to(torch.float32)

    target_dist_unsqueezed = target_dist.unsqueeze(-1)
    # -0.5 log(2π) - log(σ) - 0.5 ((y - μ) / σ)²
    log_prob = -0.5 * torch.pow((target_dist_unsqueezed - mu) / sigma, 2) - torch.log(sigma) - 0.5 * math.log(2 * math.pi)

    loss = -torch.logsumexp(log_pi + log_prob, dim=-1)

    mask = (target_dist <= dist_threshold)
    if mask.sum() > 0:
        return loss[mask].mean()
    else:
        return loss.mean()
