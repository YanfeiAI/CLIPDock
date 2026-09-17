import csv
import logging
import os
import platform
import resource
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from tqdm import tqdm

from clipdock.dock import dock_func
from clipdock.utils.constant import DATA_RAW_DIR, DATA_DOCKED_DIR
from clipdock.utils.mol import center_on_ref, read_mol, smi_gen_3d, protonate_mol, parse_docked_mol
from clipdock.utils.pocket import extract_pocket, remove_clashing_bonds
from model.embed import model_input_prepare

rdkit_logger = logging.getLogger('rdkit')
rdkit_logger.disabled = True
RDLogger.DisableLog('rdApp.*')


def normalize_suffix(suffix):
    value = suffix.strip('_') if suffix else ''
    return f'_{value}' if value else ''


class DockSet:
    def __init__(self, split, suffix=''):
        self.split = split
        self.suffix = normalize_suffix(suffix)
        self.raw_dir = DATA_RAW_DIR
        self.out_dir = os.path.join(DATA_DOCKED_DIR, self.split + self.suffix)
        self.item_list = []

        open_path = None
        if split == 'posebusters':
            self.raw_dir = os.path.join(DATA_RAW_DIR, 'posebusters')
            open_path = os.path.join(DATA_RAW_DIR, 'posebusters_test.txt')
        elif split.startswith('pdbpl'):
            self.raw_dir = os.path.join(DATA_RAW_DIR, 'pdbpl')
            open_path = os.path.join(DATA_RAW_DIR, f'{split}.txt')
        else:
            self.raw_dir = str(os.path.join(DATA_RAW_DIR, split))
        if open_path is not None and os.path.exists(open_path):
            with open(open_path, "r") as f:
                self.item_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
        else:
            self.item_list = [s for s in os.listdir(self.raw_dir) if os.path.isdir(os.path.join(self.raw_dir, s))]

    def get_item_path(self, item):
        item_dir = str(os.path.join(self.raw_dir, item))
        rec_path = os.path.join(item_dir, f'{item}_pocket.pdb')
        if not os.path.exists(rec_path):
            rec_path = os.path.join(item_dir, f'{item}_protein_pocket.pdb')
        if not os.path.exists(rec_path):
            rec_path = os.path.join(item_dir, f'{item}_protein.pdb')

        out_pocket_dir = os.path.join(DATA_DOCKED_DIR, self.split + '_pocket', item)
        if not os.path.exists(out_pocket_dir):
            os.makedirs(out_pocket_dir, exist_ok=True)
        out_pocket_path = os.path.join(str(out_pocket_dir), f'{item}_pocket.pdb')

        out_lig_dir = os.path.join(str(self.out_dir), item)
        if not os.path.exists(out_lig_dir) and len(self.suffix) > 0:
            os.makedirs(out_lig_dir, exist_ok=True)
        out_lig_path = os.path.join(str(out_lig_dir), f'{item}_docked_ligand.sdf')
        init_lig_path = os.path.join(str(out_lig_dir), f"{item}_init_ligand.sdf")

        ref_lig_path = os.path.join(item_dir, f'{item}_ligand.sdf')
        if not os.path.exists(ref_lig_path):
            ref_lig_path = os.path.join(item_dir, f'{item}_ligand.mol2')
        if not os.path.exists(ref_lig_path):
            ref_lig_path = os.path.join(item_dir, f'{item}_peptide.pdb')

        start_conf_path = os.path.join(item_dir, f"{item}_ligand_start_conf.sdf")
        if not os.path.exists(start_conf_path):
            start_conf_path = None

        return rec_path, ref_lig_path, start_conf_path, out_pocket_path, init_lig_path, out_lig_path


def process_cpx(dockset, item, remove_water=True, molwt_cutoff=500.0, rb_cutoff=10.0,
                pocket_cutoff=8.0):
    rec_path, ref_lig_path, start_conf_path, out_pocket_path, _, _ = dockset.get_item_path(item)
    ref_mol = read_mol(ref_lig_path)
    if abs(pocket_cutoff - 8.0) > 0.01:
        out_pocket_path = out_pocket_path.replace('_pocket.pdb', f'_pocket_{pocket_cutoff:.1f}.pdb')

    mol_wt = Descriptors.MolWt(Chem.RemoveHs(Chem.Mol(ref_mol)))
    rot_num = Descriptors.NumRotatableBonds(Chem.RemoveHs(Chem.Mol(ref_mol)))
    if mol_wt > molwt_cutoff or rot_num > rb_cutoff:
        return None, None
    if (dockset.split in ['train', 'val'] or dockset.split.endswith('_train') or dockset.split.endswith('_val')) and \
            'Fe' in Chem.MolToSmiles(ref_mol):
        return None, None

    if not os.path.exists(out_pocket_path):
        extract_pocket(ref_lig_path, rec_path, out_pocket_path, remove_water=remove_water,
                       threshold=pocket_cutoff)

    rec_mol = read_mol(out_pocket_path)
    if rec_mol is None:
        rec_mol = remove_clashing_bonds(out_pocket_path)

    try:
        protonate_mol(Chem.RemoveHs(Chem.Mol(rec_mol)))
    except Exception:
        print(f'remove cofactor for {item} pocket', file=sys.stderr)
        extract_pocket(ref_lig_path, rec_path, out_pocket_path, remove_water=remove_water,
                       remove_cofactor=True, threshold=pocket_cutoff)
        rec_mol = read_mol(out_pocket_path)
        if rec_mol is None:
            rec_mol = remove_clashing_bonds(out_pocket_path)

    return rec_mol, ref_mol, item


def prepare_cpx_set(split, remove_water, molwt_cutoff, rb_cutoff, pocket_cutoff):
    cpx_data = []

    dockset = DockSet(split)

    item_list = dockset.item_list
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(process_cpx, dockset, item_list[i],
                            remove_water, molwt_cutoff, rb_cutoff, pocket_cutoff): i for i in
            range(len(item_list))}
        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            try:
                res = future.result()
                if res[0] is None or res[1] is None:
                    print(f'{item_list[i]} mol is None', file=sys.stderr)
                    continue
                cpx_data.append(res)
            except Exception:
                print(f'error on {item_list[i]}', file=sys.stderr)
                continue

    print(f"cpx_num: {len(cpx_data)}", file=sys.stderr)
    if len(cpx_data) < 500:
        cpx_data = sorted(cpx_data, key=lambda x:
            (Descriptors.HeavyAtomCount(Chem.RemoveHs(Chem.Mol(x[0]))) *
             Descriptors.HeavyAtomCount(Chem.RemoveHs(Chem.Mol(x[1]))) *
             Descriptors.NumRotatableBonds(Chem.RemoveHs(Chem.Mol(x[1])))), reverse=True)

    return cpx_data


def _child_cpu_time():
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return child_usage.ru_utime + child_usage.ru_stime


def _cpu_time_snapshot():
    return time.process_time(), _child_cpu_time()


def _cpu_time_delta(start):
    process_start, child_start = start
    return (time.process_time() - process_start) + (_child_cpu_time() - child_start)


def redock(dockset, rec_mol, ref_mol, item,
           restarts=8, pose_num=32, energy_range=6.0,
           prep_model_input=False, workers=1, gscore_params_path=None):
    split = dockset.split

    rec_mol = Chem.Mol(rec_mol)
    ref_mol = Chem.Mol(ref_mol)
    rec_path, ref_lig_path, start_conf_path, out_pocket_path, init_lig_path, out_lig_path \
        = dockset.get_item_path(item)

    use_gauss = True
    if split in ['train', 'val']:
        use_gauss = False
    if split.endswith('_train') or split.endswith('_val'):
        use_gauss = False

    lig_mol_data = []
    cpu_seconds = None
    if not os.path.exists(out_lig_path):
        if start_conf_path is not None:
            input_mol = read_mol(start_conf_path)
        else:
            input_mol = smi_gen_3d(Chem.MolToSmiles(ref_mol))
        input_mol = center_on_ref(input_mol, ref_mol)

        with Chem.SDWriter(init_lig_path) as writer:
            writer.write(input_mol)

        t0 = _cpu_time_snapshot()
        dock_func(rec_mol, input_mol, rec_pred_H=True, lig_pred_H=True,
                  lig_ff_num=1, max_it=None, exhaustiveness=1,
                  restarts=restarts,
                  workers=workers, output=out_lig_path, pose_num=64,
                  energy_range=energy_range, use_gauss=use_gauss,
                  gscore_params_path=gscore_params_path)
        cpu_seconds = _cpu_time_delta(t0)

    return_pose_num = pose_num
    lig_mol_data.extend(parse_docked_mol(out_lig_path, ref_mol)[:return_pose_num])

    if prep_model_input:
        return prepare_cpx((rec_mol, lig_mol_data), pose_num, energy_range)
    return (rec_mol, lig_mol_data), cpu_seconds


def prepare_cpx(cpx, pose_num, energy_range=6.0):
    rec_mol = cpx[0]
    lig_mol_data = cpx[1][:pose_num]
    lig_mol_data = [d for d in lig_mol_data if d[1] < lig_mol_data[0][1] + energy_range]
    lig_mol_list = [d[0] for d in lig_mol_data]
    labels1 = np.array([d[-1] <= 1.0 for d in lig_mol_data], dtype=np.int8)
    labels2 = np.array([d[-1] <= 2.0 for d in lig_mol_data], dtype=np.int8)

    if labels2.sum() == 0:
        return None

    x = model_input_prepare(rec_mol, lig_mol_list)
    return x, labels1, labels2


def redock_dataset(split, suffix='', restarts=8, pose_num=32,
                   energy_range=6.0, prep_model_input=False,
                   gscore_params_path=None):
    print("prepare pocket and read mol", file=sys.stderr)
    pocket_cutoff = 8.0
    cpx_data = prepare_cpx_set(split, remove_water=True,
                               molwt_cutoff=100000, rb_cutoff=1000,
                               pocket_cutoff=pocket_cutoff)

    workers = os.cpu_count() or 1
    timing_list = []
    n_cached = 0
    n_failed = 0
    dockset = DockSet(split, suffix)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                redock, dockset, cpx_data[i][0], cpx_data[i][1], cpx_data[i][2],
                restarts, pose_num, energy_range, prep_model_input, 1,
                gscore_params_path
            ): i
            for i in range(len(cpx_data))
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            try:
                res = future.result()
                if res is not None:
                    if prep_model_input:
                        results.append(res)
                    else:
                        actual_res, cpu_seconds = res
                        results.append(actual_res)
                        if cpu_seconds is not None:
                            timing_list.append((cpx_data[i][2], cpu_seconds))
                        else:
                            n_cached += 1
            except Exception:
                n_failed += 1
                continue
    if not prep_model_input:
        _print_time_stats(timing_list, n_cached, n_failed, dockset.out_dir,
                          suffix=dockset.suffix)
    if prep_model_input:
        print(f"prepared model-input groups: {len(results)}, failed: {n_failed}", file=sys.stderr)
        return results
    return None


def _print_time_stats(timing_list, n_cached, n_failed, out_dir, suffix=''):
    from datetime import datetime
    dock_name = 'clipdock'
    timestamp = datetime.now().isoformat(timespec='seconds')
    sys_info = _get_system_info()
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not timing_list:
        n = 0
        total = mean = median = p90 = p95 = mx = 0.0
        print(f"[time stats] n=0 cached={n_cached} failed={n_failed} (no docking happened)", file=sys.stderr)
    else:
        secs = [s for _, s in timing_list]
        n = len(secs)
        total = sum(secs)
        mean = statistics.mean(secs)
        median = statistics.median(secs)
        mx = max(secs)
        if n == 1:
            p90 = p95 = mx
        else:
            p90 = statistics.quantiles(secs, n=10, method='inclusive')[-1]
            p95 = statistics.quantiles(secs, n=20, method='inclusive')[-1]
        print(f"[time stats] n={n} cached={n_cached} failed={n_failed} "
              f"sum_cpu={total:.2f}cpu·s mean={mean:.2f}cpu·s "
              f"median={median:.2f}cpu·s p90={p90:.2f}cpu·s "
              f"p95={p95:.2f}cpu·s max={mx:.2f}cpu·s", file=sys.stderr)
    save_path = os.path.join(out_dir, 'time_stats.csv')
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['item', 'cpu_seconds'])
        for item, s in timing_list:
            writer.writerow([item, f'{s:.4f}'])
    print(f"[time stats] per-item CSV saved to {save_path}", file=sys.stderr)

    summary_path = os.path.join(out_dir, 'time_stats_summary.csv')
    file_exists = os.path.exists(summary_path)
    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'suffix', 'dock',
                             'system', 'os_release', 'hostname',
                             'cpu_count', 'cpu_model', 'mem_total_gb',
                             'n', 'cached', 'failed',
                             'sum_cpu_s', 'mean_cpu_s', 'median_cpu_s',
                             'p90_cpu_s', 'p95_cpu_s', 'max_cpu_s'])
        writer.writerow([timestamp, suffix, dock_name,
                         sys_info['system'], sys_info['os_release'], sys_info['hostname'],
                         sys_info['cpu_count'], sys_info['cpu_model'], sys_info['mem_total_gb'],
                         n, n_cached, n_failed,
                         f'{total:.4f}', f'{mean:.4f}', f'{median:.4f}',
                         f'{p90:.4f}', f'{p95:.4f}', f'{mx:.4f}'])
    print(f"[time stats] summary CSV appended to {summary_path}", file=sys.stderr)


def _get_system_info():
    info = {
        'system': platform.system() or 'unknown',
        'os_release': platform.release() or 'unknown',
        'hostname': platform.node() or 'unknown',
        'cpu_count': os.cpu_count() or 0,
        'cpu_model': '',
        'mem_total_gb': 0.0,
    }
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if 'model name' in line:
                    info['cpu_model'] = line.split(':', 1)[1].strip()
                    break
    except Exception:
        pass
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    info['mem_total_gb'] = round(kb / 1024 / 1024, 2)
                    break
    except Exception:
        pass
    return info
