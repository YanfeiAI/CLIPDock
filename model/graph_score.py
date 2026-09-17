import argparse
import os

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from rdkit import Chem
from torch.utils.data import DataLoader, IterableDataset

from clipdock.utils.constant import VS_MODEL_WEIGHTS_PATH
from clipdock.utils.mol import mols_cluster
from model.graph import protein_to_graph, ligand_to_graph
from model.graph_utils import collate_graph_attributes, disable_rdkit_logging
from model.train_vs import VSTrainer

disable_rdkit_logging()

USE_AMP_GLOBAL = True


def build_parser():
    parser = argparse.ArgumentParser(
        description="Score docked ligands with the graph-based CLIPDock VS-score model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-p', '--pocket', type=str, required=True,
                        help='Path to the receptor pocket PDB file')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-l', '--ligands', type=str,
                             help='Path to an SDF file containing ligands')
    input_group.add_argument('--lig_dir', type=str,
                             help='Directory containing ligand SDF files')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Path to a custom VS-score checkpoint')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output filename prefix; .csv is appended automatically')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device index, or -1 to force CPU inference')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='Inference batch size')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='DataLoader worker processes')
    parser.add_argument('--top_dir', type=str, default=None,
                        help='Output directory prefix for the selected top molecules')
    parser.add_argument('--top_num', type=int, default=None,
                        help='Select and export this many top-scoring molecules')
    parser.add_argument('--cluster', type=float, default=0.0,
                        help='Butina clustering threshold for selected molecules; 0 disables clustering')
    return parser



class InferenceDataset(IterableDataset):
    def __init__(self, pocket_mol, source, mode='file'):
        self.pocket_mol = pocket_mol
        self.graph_rec = protein_to_graph(pocket_mol)
        self.source = source
        self.mode = mode

        if self.mode == 'dir':
            self.files = []
            for root, _, files in os.walk(source):
                for f in files:
                    if f.lower().endswith('.sdf'):
                        self.files.append(os.path.join(root, f))

    def process_mol(self, mol, idx):
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"ligand_{idx}"
        graph_ref = ligand_to_graph(mol)
        if self.graph_rec is None or graph_ref is None:
            return None

        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            smi = ""

        return {'prot': self.graph_rec, 'lig': graph_ref}, name, smi

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if self.mode == 'dir':
            files = self.files
            if worker_info is not None:
                files = files[worker_info.id::worker_info.num_workers]

            for f_path in files:
                suppl = Chem.SDMolSupplier(f_path)
                try:
                    mol = suppl[0]
                    if mol:
                        res = self.process_mol(mol, 0)
                        if res:
                            yield (*res, f_path, 0)
                except Exception:
                    pass

        else:
            suppl = Chem.SDMolSupplier(self.source)
            if worker_info is not None:
                for i, mol in enumerate(suppl):
                    if i % worker_info.num_workers == worker_info.id:
                        if mol:
                            res = self.process_mol(mol, i)
                            if res:
                                yield (*res, self.source, i)
            else:
                for i, mol in enumerate(suppl):
                    if mol:
                        res = self.process_mol(mol, i)
                        if res:
                            yield (*res, self.source, i)

def collate_graphs(batch):
    data_list = [b[0] for b in batch if b[0] is not None]
    if not data_list:
        return None, None, None, None, None

    names = [b[1] for b in batch if b[0] is not None]
    smis = [b[2] for b in batch if b[0] is not None]
    f_paths = [b[3] for b in batch if b[0] is not None]
    m_indices = [b[4] for b in batch if b[0] is not None]

    prot_batch = collate_graph_attributes(data_list, 'prot')
    lig_batch = collate_graph_attributes(data_list, 'lig')

    return (prot_batch, lig_batch), names, smis, f_paths, m_indices

def compute_mdn_scores_batch(model, batch_data, device):
    prot_batch, lig_batch = batch_data

    p_x, p_idx, p_attr, p_b = prot_batch['x'].to(device), prot_batch['edge_index'].to(device), prot_batch['edge_attr'].to(device), prot_batch['batch'].to(device)
    l_x, l_idx, l_attr, l_b = lig_batch['x'].to(device), lig_batch['edge_index'].to(device), lig_batch['edge_attr'].to(device), lig_batch['batch'].to(device)

    with torch.no_grad():
        with torch.amp.autocast(
                device_type='cuda',
                enabled=USE_AMP_GLOBAL and device.type == 'cuda',
                dtype=torch.float16):
            result = model.model(p_x, p_idx, p_attr, p_b, l_x, l_idx, l_attr, l_b)
        log_pi, mu, sigma, pair_counts = result

        if log_pi is None:
            return [None] * len(pair_counts)

        log_pi = log_pi.to(torch.float32)
        mu = mu.to(torch.float32)
        sigma = sigma.to(torch.float32)

        prot_pos = prot_batch['pos'].cpu().numpy()
        lig_pos = lig_batch['pos'].cpu().numpy()

        p_counts = torch.bincount(p_b).cpu().numpy()
        l_counts = torch.bincount(l_b).cpu().numpy()
        p_starts = np.insert(np.cumsum(p_counts), 0, 0)
        l_starts = np.insert(np.cumsum(l_counts), 0, 0)

        all_target_dists = []
        for i in range(len(p_counts)):
            s_p_pos = prot_pos[p_starts[i] : p_starts[i+1]]
            s_l_pos = lig_pos[l_starts[i] : l_starts[i+1]]
            dist_mat = np.linalg.norm(s_p_pos[:, None, :] - s_l_pos[None, :, :], axis=2)
            all_target_dists.append(dist_mat.flatten())

        target_dist = torch.tensor(np.concatenate(all_target_dists), dtype=torch.float, device=device).unsqueeze(-1)

        log_prob = -0.5 * torch.pow((target_dist - mu) / sigma, 2) - torch.log(sigma) - 0.9189385332046727
        pair_prob = torch.exp(torch.logsumexp(log_pi + log_prob, dim=-1))

        dists = target_dist.squeeze(-1)
        pair_prob = pair_prob / (torch.pow(dists, 2) + 1e-6)

        scores = [p.sum().item() for p in torch.split(pair_prob, pair_counts)]
    return scores

def check_existing_results(output_csv):
    if os.path.exists(output_csv):
        try:
            df = pd.read_csv(output_csv)
            if {'name', 'score'}.issubset(df.columns):
                print(f"Found existing results at {output_csv}, skipping inference.")
                return df
            print(f"Existing results at {output_csv} are missing minimal columns. Re-running inference.")
        except Exception as e:
            print(f"Error reading existing results: {e}. Re-running inference.")
    return None

def run_inference_logic(args, device):
    model = VSTrainer.load_from_checkpoint(args.model, map_location=device)
    model.eval()

    pocket_mol = Chem.MolFromPDBFile(args.pocket)
    if pocket_mol is None:
        print(f"Error: Could not read pocket from {args.pocket}")
        return pd.DataFrame()

    if args.ligands:
        dataset = InferenceDataset(pocket_mol, args.ligands, mode='file')
    elif args.lig_dir:
        dataset = InferenceDataset(pocket_mol, args.lig_dir, mode='dir')
    else:
        raise ValueError("Either --ligands or --lig_dir must be provided")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_graphs)

    results = []
    for batch_data, batch_names, batch_smis, batch_fpaths, batch_indices in tqdm(dataloader, desc="Scoring", unit="batch"):
        if batch_data is None:
            continue
        scores = compute_mdn_scores_batch(model, batch_data, device)
        for name, score, smi, f_path, m_idx in zip(batch_names, scores, batch_smis, batch_fpaths, batch_indices):
            if score is not None:
                results.append({"name": name, "score": score, "smi": smi, "f_path": f_path, "m_idx": m_idx})

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('score', ascending=False).drop_duplicates('name')
    return df

def find_molecules(top_df, args):
    top_mols = [None] * len(top_df)

    if 'f_path' in top_df.columns and 'm_idx' in top_df.columns:
        fpath_to_indices = defaultdict(list)
        for i, (_, row) in enumerate(top_df.iterrows()):
            fpath_to_indices[row['f_path']].append((row['m_idx'], i))

        for f_path, idx_list in tqdm(fpath_to_indices.items(), desc="Loading top mols"):
            try:
                suppl = Chem.SDMolSupplier(f_path)
                for m_idx, original_pos in idx_list:
                    try:
                        mol = suppl[m_idx]
                        top_mols[original_pos] = mol
                    except Exception:
                        pass
            except Exception:
                pass
        return top_mols

    target_names = {row['name']: i for i, (_, row) in enumerate(top_df.iterrows())}
    found_count = 0

    if args.lig_dir:
        print("Attempting direct file lookup by name...")
        for name, idx in target_names.items():
            if top_mols[idx] is not None:
                continue

            potential_paths = [
                os.path.join(args.lig_dir, f"{name}.sdf"),
                os.path.join(args.lig_dir, name)
            ]

            for p in potential_paths:
                if os.path.exists(p):
                    try:
                        suppl = Chem.SDMolSupplier(p)
                        if len(suppl) > 0 and suppl[0]:
                            top_mols[idx] = suppl[0]
                            found_count += 1
                            break
                    except Exception:
                        pass

    files_to_scan = []
    if found_count < len(target_names):
        if args.ligands:
            files_to_scan.append(args.ligands)
        elif args.lig_dir:
            print("Direct lookup incomplete, scanning directory...")
            for root, _, files in os.walk(args.lig_dir):
                for f in files:
                    if f.lower().endswith('.sdf'):
                        files_to_scan.append(os.path.join(root, f))

    for f_path in tqdm(files_to_scan, desc="Scanning mols"):
        if found_count >= len(target_names):
            break
        try:
            suppl = Chem.SDMolSupplier(f_path)
            for mol in suppl:
                if mol and mol.HasProp("_Name"):
                    name = mol.GetProp("_Name")
                    if name in target_names:
                        idx = target_names[name]
                        if top_mols[idx] is None:
                            top_mols[idx] = mol
                            found_count += 1
        except Exception:
            pass

    return top_mols

def process_top_results(df, args):
    if args.top_num is None:
        return

    df_sorted = df.sort_values('score', ascending=False)
    df_unique = df_sorted.drop_duplicates('name')
    top_df = df_unique.head(args.top_num).copy()

    top_df['mol'] = find_molecules(top_df, args)

    if 'smi' not in top_df.columns:
        top_df['smi'] = ""

    smi_list = []
    for mol, existing_smi in zip(top_df['mol'], top_df['smi']):
        if mol:
            try:
                smi_list.append(Chem.MolToSmiles(mol))
            except Exception:
                smi_list.append(existing_smi if pd.notna(existing_smi) else "")
        else:
            smi_list.append(existing_smi if pd.notna(existing_smi) else "")
    top_df['smi'] = smi_list

    out_name = f"{args.output}_top{args.top_num}"
    out_dir = args.top_dir

    if args.cluster > 0.0:
        mols = top_df['mol'].tolist()
        if mols:
            clusters = mols_cluster(mols, threshold=args.cluster)
            keep_indices = [min(c) for c in clusters]
            top_df = top_df.iloc[keep_indices].copy()

        out_name += f"_cluster{args.cluster}"
        if out_dir:
            out_dir += f"_cluster{args.cluster}"

    top_df = top_df.sort_values('score', ascending=False)
    top_df['rank'] = range(1, len(top_df) + 1)
    final_cols = ['rank', 'name', 'score', 'smi']

    top_df[final_cols].to_csv(out_name + ".csv", index=False)
    print(f"Saved top results to {out_name}.csv")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for i, (_, row) in enumerate(top_df.iterrows()):
            mol = row['mol']
            name = row['name']
            if mol:
                mol.SetProp("_Name", name)
                safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '.', '_', '-')]).strip()
                if not safe_name:
                    safe_name = f"mol_{i+1}"

                writer = Chem.SDWriter(os.path.join(out_dir, f"{i+1:04d}_{safe_name}.sdf"))
                writer.write(mol)
                writer.close()
        print(f"Saved top molecules to {out_dir}")

def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output_csv = args.output + ".csv"
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df = check_existing_results(output_csv)

    if df is None:
        args.model = os.path.abspath(os.fspath(args.model or VS_MODEL_WEIGHTS_PATH))
        if not os.path.isfile(args.model):
            raise FileNotFoundError(
                f"VS-score checkpoint does not exist: {args.model}. "
                "Download vs_model.ckpt from GitHub Releases and place it in the model directory, "
                "or pass --model."
            )
        print(f"Using VS-score checkpoint: {args.model}")
        df = run_inference_logic(args, device)
        if not df.empty:
            df[['name', 'score']].to_csv(output_csv, index=False)
            print(f"Saved results to {args.output}.csv")

    if not df.empty and args.top_num is not None:
        process_top_results(df, args)


def main(argv=None):
    run(build_parser().parse_args(argv))

if __name__ == '__main__':
    main()
