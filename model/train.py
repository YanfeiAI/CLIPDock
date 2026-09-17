import logging
import os
import pickle
import random
import argparse
import math
from pathlib import Path
import numpy as np

from rdkit import RDLogger
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from clipdock.atom_type import GscoreAtomType
from clipdock.utils.constant import DATA_DOCKED_DIR, DATA_ROOT
from model.model import GScoreModel
from model.redock import normalize_suffix, redock_dataset

rdkit_logger = logging.getLogger('rdkit')
rdkit_logger.disabled = True
RDLogger.DisableLog('rdApp.*')

def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the G-score model and export decoded Gaussian parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", type=Path, default=Path('data/training/gscore'),
                        help="Training output directory")
    parser.add_argument("--data_suffix", type=str, required=True,
                        help="Name appended to cached docking datasets, for example r8p32")
    parser.add_argument("--devices", type=str, default=None,
                        help="CUDA device IDs passed through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--epoch", "--epochs", dest="epoch", type=int, default=5,
                        help="Number of G-score training epochs")
    parser.add_argument("--lr", type=float, default=0.00003,
                        help="Initial AdamW learning rate")
    parser.add_argument("--restarts", type=int, default=8,
                        help="Independent docking restarts used to generate training poses")
    parser.add_argument("--pose_num", type=int, default=32,
                        help="Maximum number of poses retained per complex")
    parser.add_argument("--energy_range", type=float, default=6.0,
                        help="Maximum E-score range used when retaining poses")
    parser.add_argument("--refresh", action='store_true',
                        help="Regenerate cached training and validation datasets")
    parser.add_argument("--train_split", type=str, default='pdbpl_train',
                        help="Training split name; reads data/raw/<split>.txt")
    parser.add_argument("--val_split", type=str, default='pdbpl_val',
                        help="Validation split name; reads data/raw/<split>.txt")
    return parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8
        self.label1_margin = 0.2
        self.label2_margin = 0.2

    def forward(self, logits: torch.Tensor, labels1: torch.Tensor, labels2: torch.Tensor):
        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        logits = F.softmax(-logits, dim=-1)
        label1_margin = self.label1_margin
        label2_margin = self.label2_margin

        pos1_mask = labels1 == 1
        neg1_mask = labels1 == 0
        if pos1_mask.any() and neg1_mask.any():
            pos1_logits = logits[pos1_mask]
            neg1_logits = logits[neg1_mask]
            differences1 = neg1_logits.unsqueeze(0) - pos1_logits.unsqueeze(1)
            loss = loss + F.relu(differences1 + label1_margin).mean()

        pos2_mask = labels2 == 1
        neg2_mask = labels2 == 0
        if pos2_mask.any() and neg2_mask.any():
            pos2_logits = logits[pos2_mask]
            neg2_logits = logits[neg2_mask]
            differences2 = neg2_logits.unsqueeze(0) - pos2_logits.unsqueeze(1)
            loss = loss + F.relu(differences2 + label2_margin).mean()

        return loss



def get_dataset(split, suffix, restarts=8, pose_num=32, energy_range=6.0, refresh=False):
    suffix = normalize_suffix(suffix)
    pkl_path = f"{DATA_DOCKED_DIR}/{split}{suffix}_dataset.pkl"
    if refresh:
        if os.path.exists(pkl_path):
            print('remove:', pkl_path)
            os.remove(pkl_path)
    if not refresh and os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = redock_dataset(split=split, suffix=suffix,
                              restarts=restarts, pose_num=pose_num,
                              energy_range=energy_range,
                              prep_model_input=True)
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)

    return data


def val_model(model, val_results, criterion):
    model.eval()
    val_loss_total = 0.0

    with torch.no_grad():
        for i, (x, labels1, labels2) in enumerate(val_results):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            labels1 = torch.tensor(labels1, dtype=torch.float32).to(device)
            labels2 = torch.tensor(labels2, dtype=torch.float32).to(device)

            outputs = model(x)
            loss = criterion(outputs[:, 0], labels1, labels2)
            val_loss_total += loss.item()

    val_loss_total /= max(len(val_results), 1)
    return val_loss_total


def report_dataset_labels(name, results):
    mixed_1a = 0
    mixed_2a = 0
    for _, labels1, labels2 in results:
        mixed_1a += int(np.any(labels1 == 1) and np.any(labels1 == 0))
        mixed_2a += int(np.any(labels2 == 1) and np.any(labels2 == 0))
    informative = sum(
        1 for _, labels1, labels2 in results
        if ((np.any(labels1 == 1) and np.any(labels1 == 0)) or
            (np.any(labels2 == 1) and np.any(labels2 == 0)))
    )
    print(f'{name} dataset: groups={len(results)}, mixed@1A={mixed_1a}, '
          f'mixed@2A={mixed_2a}, informative={informative}')
    if not results or informative == 0:
        raise RuntimeError(
            f'{name} dataset has no informative positive/negative pose groups; '
            'refusing to train'
        )


def train(epochs=5, init_lr=0.00003, suffix=None, output_dir=None,
          restarts=8, pose_num=32, energy_range=6.0,
          train_split='pdbpl_train', val_split='pdbpl_val',
          refresh=False):
    suffix = normalize_suffix(suffix)
    if not suffix:
        raise ValueError("data suffix must not be empty")
    output_dir = Path(output_dir or Path(DATA_ROOT) / 'training/gscore').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / 'model.pth'

    criterion = ContrastiveLoss()
    val_result = get_dataset(val_split, suffix, restarts=restarts, pose_num=pose_num,
                             energy_range=energy_range, refresh=refresh)
    report_dataset_labels('validation', val_result)
    val_loss_total = val_model(GScoreModel().to(device), val_result, criterion)
    print(f'Init | Val | tot_loss: {val_loss_total:.4f}')
    train_result = get_dataset(train_split, suffix, restarts=restarts, pose_num=pose_num,
                               energy_range=energy_range, refresh=refresh)
    report_dataset_labels('training', train_result)
    train_len = len(train_result)

    print('prepared_input')
    print(f"train: {len(train_result)}")
    print(f"val: {len(val_result)}")

    model = GScoreModel().to(device)
    print(f"device: {next(model.parameters()).device}")
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-2)
    tot_steps = epochs * train_len
    warmup_steps = 0.03 * tot_steps
    end_lr_ratio = 0.1
    scheduler = LambdaLR(optimizer, lr_lambda=lambda steps: steps / warmup_steps if steps < warmup_steps else \
        end_lr_ratio + (1 - end_lr_ratio) * 0.5 *
        (1.0 + math.cos(float(steps - warmup_steps) / max(1, tot_steps - warmup_steps) * math.pi)))

    best_epoch = 0
    best_val_loss = 1e8

    print('training...')
    print(f"init_lr: {init_lr:.6f}, epochs: {epochs}")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        epoch_data = train_result
        random.shuffle(epoch_data)
        for i, (x, labels1, labels2) in tqdm(enumerate(epoch_data), total=len(epoch_data)):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            labels1 = torch.tensor(labels1, dtype=torch.float32).to(device)
            labels2 = torch.tensor(labels2, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs[:, 0], labels1, labels2)
            loss.backward()

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= max(train_len, 1)

        val_loss_total = 999
        if (epoch + 1) % (max(epochs // epochs, 1)) == 0 or epoch == epochs - 1:
            val_loss_total = val_model(model, val_result, criterion)
            if val_loss_total < best_val_loss:
                best_val_loss = val_loss_total
                best_epoch = epoch + 1
                if save_path is not None:
                    model.eval()
                    print(f'Save model on epoch{epoch + 1}')
                    torch.save(model.state_dict(), save_path)
        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss_total:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

    print(f'best_val_loss: {best_val_loss:.4f}, best_epoch: {best_epoch}')
    print(f"model saved to {save_path}")

    if not save_path.is_file():
        raise RuntimeError("Best-validation checkpoint was not saved; cannot export G-score")
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"loaded best-validation checkpoint from epoch {best_epoch} for G-score export")
    model.eval()
    try:
        param = model.decode()
        param = param.cpu().detach().numpy()
        type_class = GscoreAtomType()

        param_str = f"lig,rec,{','.join(['g' + str(i+1) for i in range(param.shape[-1])])}\n"
        for i in range(len(type_class.type_list)):
            for j in range(len(type_class.type_list)):
                param_str += f"{type_class.type_list[i]},{type_class.type_list[j]}," + ",".join([f"{w:.4g}" for w in param[i, j, :]]) + '\n'
        save_gauss_path = output_dir / 'gscore.csv'
        with open(save_gauss_path, "w") as f:
            f.write(param_str)
        print(f"model_gauss saved to {save_gauss_path}")
    except Exception as exc:
        raise RuntimeError("Failed to export G-score from the best-validation checkpoint") from exc


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(epochs=args.epoch, init_lr=args.lr, suffix=args.data_suffix, output_dir=args.output_dir,
          restarts=args.restarts, pose_num=args.pose_num, energy_range=args.energy_range,
          train_split=args.train_split, val_split=args.val_split,
          refresh=args.refresh)


if __name__ == '__main__':
    main()
