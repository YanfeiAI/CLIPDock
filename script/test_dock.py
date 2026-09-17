import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from model.redock import DockSet, redock_dataset
from clipdock.utils.mol import read_mol, parse_docked_mol

import logging

rdkit_logger = logging.getLogger('rdkit')
rdkit_logger.disabled = True

logging.getLogger("posebusters").setLevel(logging.ERROR)
try:
    from posebusters import PoseBusters
except ImportError:
    pass

METRIC_HEADER = (
    'dataset', 'method', 'repeat',
    'top1_rmsd<1', 'top1_rmsd<1_valid', 'top1_rmsd<2', 'top1_rmsd<2_valid',
    'top1_rmsd_median', 'top1_rmsd_mean',
    'top5_rmsd<1', 'top5_rmsd<2', 'top5_rmsd_median', 'top5_rmsd_mean',
    'best_rmsd<1', 'best_rmsd<2', 'success/tot_num',
)


def bust_path(mol_pred_path, mol_true_path, mol_cond_path):
    p = PoseBusters("redock")
    df = p.bust([mol_pred_path], mol_true_path, mol_cond_path)
    return df.all(axis=1).values[0]


def eval_cpx(rec_path, docked_path, ref_path, bust=True):
    ref_mol = read_mol(ref_path)
    if ref_mol is None:
        raise Exception("ref_mol is None")
    lig_mol_data = parse_docked_mol(docked_path, ref_mol)

    if lig_mol_data[0][-1] > 98:
        raise Exception("lig_mol_data[0][-1] > 98")

    top1_path = docked_path.replace('.sdf', '_top1.sdf')
    with Chem.SDWriter(top1_path) as writer:
        writer.write(lig_mol_data[0][0])
    top1_rmsd_valid = False
    if bust and lig_mol_data[0][-1] < 2.0:
        try:
            top1_rmsd_valid = bust_path(top1_path, ref_path, rec_path)
        except Exception:
            pass

    result = {}
    result['item'] = os.path.basename(ref_path).replace('_ligand.sdf', '')
    result['top1_rmsd'] = lig_mol_data[0][-1]
    result['top1_rmsd<1'] = lig_mol_data[0][-1] < 1
    result['top1_rmsd<1&valid'] = lig_mol_data[0][-1] < 1 and top1_rmsd_valid
    result['top1_rmsd<2'] = lig_mol_data[0][-1] < 2
    result['top1_rmsd<2&valid'] = lig_mol_data[0][-1] < 2 and top1_rmsd_valid
    result['top5_rmsd'] = min(lig_mol_data[i][-1] for i in range(min(5, len(lig_mol_data))))
    result['top5_rmsd<1'] = any(lig_mol_data[i][-1] < 1 for i in range(min(5, len(lig_mol_data))))
    result['top5_rmsd<2'] = any(lig_mol_data[i][-1] < 2 for i in range(min(5, len(lig_mol_data))))
    result['best_rmsd<1'] = any(lig_mol_data[i][-1] < 1 for i in range(len(lig_mol_data)))
    result['best_rmsd<2'] = any(lig_mol_data[i][-1] < 2 for i in range(len(lig_mol_data)))
    return result


def eval_set(split, suffix='', cpx_tot_num=None, bust=True, save_csv=False):
    dockset = DockSet(split, suffix=suffix)
    item_list = dockset.item_list

    item_to_prop = {}

    for item in item_list:
        out_lig_path = dockset.get_item_path(item)[5]
        mol = read_mol(out_lig_path)
        if mol is None:
            continue
        item_to_prop[item] = {
            'molwt': Descriptors.MolWt(mol),
            'heavy_atom': Descriptors.HeavyAtomCount(mol),
            'rot_num': Descriptors.NumRotatableBonds(mol)
        }

    rec_path_list = [dockset.get_item_path(item)[3] for item in item_list]
    docked_path_list = [dockset.get_item_path(item)[5] for item in item_list]
    sdf_ref_path_list = [dockset.get_item_path(item)[1] for item in item_list]
    ref_path_list = []
    for i, path in enumerate(sdf_ref_path_list):
        if path.endswith('.sdf'):
            try:
                mol = Chem.SDMolSupplier(path)[0]
            except Exception:
                mol = None
            if mol is None:
                path = os.path.join(os.path.dirname(path), f"{item_list[i]}_ligand.mol2")
        ref_path_list.append(path)

    result_list = []
    res_list = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(eval_cpx, rec_path_list[i],
                            docked_path_list[i], ref_path_list[i], bust): i
            for i in range(len(item_list))}
        for future in as_completed(futures):
            i = futures[future]
            try:
                res = future.result()
                result_list.append(res)
                res_list.append(res)
            except Exception:
                continue
    if cpx_tot_num is None:
        cpx_tot_num = len(result_list)

    top1_rmsd_list = sorted([res['top1_rmsd'] for res in result_list])
    top1_rmsd_less1 = sum(res['top1_rmsd<1'] for res in result_list) / cpx_tot_num
    top1_rmsd_less1_valid = sum(res['top1_rmsd<1&valid'] for res in result_list) / cpx_tot_num
    top1_rmsd_less2 = sum(res['top1_rmsd<2'] for res in result_list) / cpx_tot_num
    top1_rmsd_less2_valid = sum(res['top1_rmsd<2&valid'] for res in result_list) / cpx_tot_num
    top5_rmsd_list = sorted([res['top5_rmsd'] for res in result_list])
    top5_rmsd_less1 = sum(res['top5_rmsd<1'] for res in result_list) / cpx_tot_num
    top5_rmsd_less2 = sum(res['top5_rmsd<2'] for res in result_list) / cpx_tot_num
    best_rmsd_less1 = sum(res['best_rmsd<1'] for res in result_list) / cpx_tot_num
    best_rmsd_less2 = sum(res['best_rmsd<2'] for res in result_list) / cpx_tot_num
    top1_rmsd_median = np.median(np.array(top1_rmsd_list))
    top5_rmsd_median = np.median(np.array(top5_rmsd_list))
    top1_rmsd_mean = np.mean(np.array(top1_rmsd_list))
    top5_rmsd_mean = np.mean(np.array(top5_rmsd_list))
    res_str = (f"{top1_rmsd_less1:.4f},"
               f"{top1_rmsd_less1_valid:.4f},"
               f"{top1_rmsd_less2:.4f},"
               f"{top1_rmsd_less2_valid:.4f},"
               f"{top1_rmsd_median:.2f},{top1_rmsd_mean:.2f},"
               f"{top5_rmsd_less1:.4f},{top5_rmsd_less2:.4f},"
               f"{top5_rmsd_median:.2f},{top5_rmsd_mean:.2f},"
               f"{best_rmsd_less1:.4f},{best_rmsd_less2:.4f},"
               f"{len(result_list)}/{cpx_tot_num}")
    if save_csv:
        save_path = os.path.join(str(dockset.out_dir), 'rmsd_info.csv')
        with open(save_path, 'w') as write_f:
            writer = csv.writer(write_f)
            writer.writerow(['item', 'top1 rmsd', 'top1 rmsd<2Å&valid', 'top5 rmsd',
                             'molwt', 'heavy_atom', 'rot_num'])
            for res in res_list:
                writer.writerow([res['item'], res['top1_rmsd'],
                                 res['top1_rmsd<2&valid'], res['top5_rmsd'],
                                 item_to_prop[res['item']]['molwt'],
                                 item_to_prop[res['item']]['heavy_atom'],
                                 item_to_prop[res['item']]['rot_num']
                                 ])
    return res_str


def format_metric_row(dataset, method, repeat, metrics):
    """Add the identifying columns to the metric-only result from eval_set."""
    return ','.join(str(value) for value in metric_fields(dataset, method, repeat, metrics))


def metric_fields(dataset, method, repeat, metrics):
    return [dataset, method, repeat, *metrics.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Dock and evaluate a CLIPDock benchmark split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", type=str, required=True,
                        help="Benchmark split name, for example posebusters or posex_sd")
    parser.add_argument("--suffix", type=str, default='',
                        help="Suffix used for docking outputs and cached results")
    parser.add_argument("--restarts", type=int, default=32,
                        help="Independent docking restarts per complex")
    parser.add_argument("--pose_num", type=int, default=32,
                        help="Maximum number of poses retained per complex")
    parser.add_argument("--cpx_num", type=int, default=None,
                        help="Expected total complex count used as the evaluation denominator; does not limit docking")
    parser.add_argument("--repeat", type=int, default=None,
                        help="Repeat the docking/evaluation run this many times")
    parser.add_argument("--bust", action='store_true',
                        help="Run PoseBusters validation for qualifying top poses")
    parser.add_argument("--eval_only", action='store_true',
                        help="Skip docking and evaluate existing output files")
    parser.add_argument("--metrics_output", type=str, default=None,
                        help="Write summary metrics directly to this CSV file")
    parser.add_argument("--gscore", type=str, default=None,
                        help="Path to the exported G-score parameter CSV")
    args = parser.parse_args()

    if args.gscore is not None:
        args.gscore = os.path.abspath(args.gscore)
        if not os.path.isfile(args.gscore):
            parser.error(f"G-score parameter file does not exist: {args.gscore}")

    cpx_tot_num = args.cpx_num

    suffix_list = []
    if args.repeat is not None:
        for i in range(args.repeat):
            suffix_list.append(args.suffix + f'_rp{i + 1}')
    else:
        suffix_list.append(args.suffix)

    metrics_file = None
    metrics_writer = None
    metrics_path = None
    try:
        if args.metrics_output is not None:
            metrics_path = os.path.abspath(args.metrics_output)
            os.makedirs(os.path.dirname(metrics_path) or '.', exist_ok=True)
            metrics_file = open(metrics_path, 'w', newline='', encoding='utf-8')
            metrics_writer = csv.writer(metrics_file)

        if not args.eval_only:
            for suffix in suffix_list:
                redock_dataset(args.split, suffix, args.restarts, args.pose_num,
                               gscore_params_path=args.gscore)

        if args.split not in ['train', 'val']:
            if metrics_writer is None:
                print(','.join(METRIC_HEADER))
            else:
                metrics_writer.writerow(METRIC_HEADER)
                metrics_file.flush()

            method = 'clipdock'
            for repeat, suffix in enumerate(suffix_list, start=1):
                res_str = eval_set(split=args.split, suffix=suffix, cpx_tot_num=cpx_tot_num, bust=args.bust,
                                   save_csv=True
                                   )
                row = metric_fields(args.split, method, repeat, res_str)
                if metrics_writer is None:
                    print(format_metric_row(args.split, method, repeat, res_str))
                else:
                    metrics_writer.writerow(row)
                    metrics_file.flush()
    finally:
        if metrics_file is not None:
            metrics_file.close()
        if metrics_path is not None:
            print(f"Metrics written to {metrics_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
