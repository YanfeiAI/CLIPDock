import os
import pickle
import torch
import numpy as np
import gc
from torch.utils.data import DataLoader, Dataset
from model.graph import protein_to_graph, ligand_to_graph
from model.redock import DockSet, normalize_suffix, prepare_cpx_set
from clipdock.utils.mol import parse_docked_mol
from clipdock.utils.constant import DATA_DOCKED_DIR
from model.graph_utils import to_numpy, to_tensor, collate_graph_attributes
import pytorch_lightning as pl
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_item(args):
    cpx_item, split_name, data_suffix = args
    rec_mol, ref_mol, item_id = cpx_item

    try:
        dockset = DockSet(split_name, data_suffix)
        _, _, _, _, _, out_lig_path = dockset.get_item_path(item_id)
        if os.path.exists(out_lig_path):
            lig_mol_data = parse_docked_mol(out_lig_path, ref_mol)
            if lig_mol_data:
                best_pose = min(lig_mol_data, key=lambda x: x[2])
                if best_pose[2] <= 2.0:
                    ref_mol = best_pose[0]
    except Exception:
        pass

    try:
        graph_rec = protein_to_graph(rec_mol)
        graph_ref = ligand_to_graph(ref_mol)

        if graph_rec is None or graph_ref is None:
            return None

        p_pos = graph_rec['pos']
        l_pos = graph_ref['pos']

        dists = np.linalg.norm(p_pos[:, None, :] - l_pos[None, :, :], axis=2)
        target_dist = dists.flatten().astype(np.float32)

        return {
            'prot': to_numpy(graph_rec),
            'lig': to_numpy(graph_ref),
            'target_dist': target_dist
        }
    except Exception as e:
        print(f"Error processing {item_id}: {e}")
        return None

class MolGraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = [to_tensor(d) for d in data_list if d is not None]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

class MoleculeDataModule(pl.LightningDataModule):
    def __init__(self, train_split='pdbpl_train', val_split='pdbpl_val',
                 batch_size=32, num_workers=8, pocket_cutoff=8.0,
                 data_suffix=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.pocket_cutoff = pocket_cutoff
        self.data_suffix = normalize_suffix(data_suffix)
        if not self.data_suffix:
            raise ValueError("data suffix must not be empty")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            val_data = self.load_data(self.val_split)
            self.val_dataset = MolGraphDataset(val_data)
            del val_data
            gc.collect()

            train_data = self.load_data(self.train_split)
            self.train_dataset = MolGraphDataset(train_data)
            del train_data
            gc.collect()

    def load_data(self, split):
        pocket_suffix = ''
        if abs(self.pocket_cutoff - 8.0) > 0.01:
            pocket_suffix = f'_pocket{self.pocket_cutoff:.1f}'
        cache_path = os.path.join(
            DATA_DOCKED_DIR,
            f"{split}{self.data_suffix}_vs{pocket_suffix}.pkl",
        )
        legacy_cache_path = os.path.join(
            DATA_DOCKED_DIR,
            f"{split}{self.data_suffix}_pose_dataset_vsa{pocket_suffix}.pkl",
        )
        for candidate_path in (cache_path, legacy_cache_path):
            if not os.path.exists(candidate_path):
                continue
            print(f"Loading cached data from {candidate_path}")
            try:
                with open(candidate_path, 'rb') as f:
                    data_list = pickle.load(f)
                if candidate_path != cache_path:
                    try:
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        with open(cache_path, 'wb') as f:
                            pickle.dump(data_list, f)
                    except Exception as e:
                        print(f"Failed to migrate cache: {e}")
                return data_list
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-processing...")

        cpx_set = prepare_cpx_set(split, remove_water=True,
                                  molwt_cutoff=1500, rb_cutoff=1000,
                                  pocket_cutoff=self.pocket_cutoff)

        task_args = [(item, split, self.data_suffix) for item in cpx_set]

        with ProcessPoolExecutor(max_workers=max(self.num_workers, 1)) as executor:
            data_list = list(tqdm(executor.map(process_item, task_args, chunksize=10), total=len(cpx_set), desc=f"Precomputing {split} Graphs"))

        del cpx_set
        del task_args
        gc.collect()

        data_list = [d for d in data_list if d is not None]

        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(data_list, f)
            print(f"Saved data cache to {cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

        return data_list

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=self.collate_fn,
                          pin_memory=True, persistent_workers=(self.num_workers > 0))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=self.collate_fn,
                          pin_memory=True, persistent_workers=(self.num_workers > 0))

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        prot_batch = collate_graph_attributes(batch, 'prot')
        lig_batch = collate_graph_attributes(batch, 'lig')
        target_dist = torch.cat([g['target_dist'] for g in batch], dim=0)

        return prot_batch, lig_batch, target_dist
