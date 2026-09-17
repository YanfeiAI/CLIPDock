import csv
import json
import logging
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdkit import RDLogger, Chem
from tqdm import tqdm

from clipdock.dock import dock_func, setup_logging
from clipdock.utils.constant import GAUSS_PARAMS_PATH, VERSION
from clipdock.utils.mol import read_mol, center_on_ref, center_on_coords, smi_gen_3d, parse_docked_mol
from clipdock.utils.pocket import extract_pocket, read_coords

rdkit_logger = logging.getLogger('rdkit')
rdkit_logger.disabled = True
RDLogger.DisableLog('rdApp.*')
MAX_SCORE = 9999

clipdock_logger = logging.getLogger('clipdock')


def safe_json_load(filepath, default=None):
    if default is None:
        default = {}
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        clipdock_logger.warning(f"Failed to load {filepath}: {e}, using default")
    return default


def safe_json_save(data, filepath):
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(filepath), delete=False) as tmp_file:
            json.dump(data, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        shutil.move(tmp_file.name, filepath)
        return True
    except (IOError, TypeError, OSError) as e:
        clipdock_logger.error(f"Failed to save {filepath}: {e}")
        try:
            if os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)
        except Exception:
            pass
        return False


def save_csv(results, output_dir, filename, sort_results=True):
    if sort_results:
        results = sorted(results, key=lambda x: x[1])

    csv_path = os.path.join(output_dir, filename)
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', dir=output_dir, delete=False, suffix='.csv') as tmp_file:
            writer = csv.writer(tmp_file)
            writer.writerow(['name', 'CLIPDock score'])
            for name, score in results:
                writer.writerow([name, f"{score:.5f}"])
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        shutil.move(tmp_file.name, csv_path)
    except Exception as e:
        clipdock_logger.error(f"Failed to save CSV {csv_path}: {e}")
        try:
            if os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)
        except Exception:
            pass


def generate_top_files(layer_results, output_dir, layer_dir, prefix_name, top_num=300):
    top_dir = os.path.join(output_dir, f"{prefix_name}_top{top_num}")
    os.makedirs(top_dir, exist_ok=True)

    top_result = sorted(layer_results, key=lambda x: x[1])[:top_num]

    top_csv = os.path.join(top_dir, f"top{top_num}_list.csv")
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', dir=top_dir, delete=False, suffix='.csv') as tmp_file:
            writer = csv.writer(tmp_file)
            writer.writerow(['rank', 'name', 'CLIPDock score', 'smi'])
            for i, (name, score) in enumerate(top_result, 1):
                sdf_file = os.path.join(layer_dir, f"{name}.sdf")
                mol = read_mol(sdf_file)
                if mol is None:
                    smi = ''
                else:
                    smi = Chem.MolToSmiles(mol, canonical=True)
                writer.writerow([i, name, f"{score:.5f}", smi])
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        shutil.move(tmp_file.name, top_csv)
    except Exception as e:
        clipdock_logger.error(f"Failed to save top CSV {top_csv}: {e}")
        try:
            if os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)
        except Exception:
            pass

    clipdock_logger.info(f"Generated top{top_num} list with {len(top_result)} molecules")

    sdf_dir = os.path.join(top_dir, "sdf_files")
    os.makedirs(sdf_dir, exist_ok=True)

    clipdock_logger.info(f"Copying SDF files for top{top_num} molecules...")
    copied_count = 0

    for rank, (name, score) in enumerate(top_result, 1):
        source_sdf = os.path.join(layer_dir, f"{name}.sdf")
        if os.path.exists(source_sdf):
            target_sdf = os.path.join(sdf_dir, f"{rank:03d}_{name}.sdf")
            try:
                shutil.copy2(source_sdf, target_sdf)
                copied_count += 1
            except Exception as e:
                clipdock_logger.error(f"Failed to copy SDF {source_sdf} to {target_sdf}: {e}")

    clipdock_logger.info(f"Successfully copied {copied_count} SDF files to {sdf_dir}")


def run_clipdock(rec_mol, ref_mol, input_name_smi, restarts, out_dir, pose_num, pad_size,
                 gscore_params_path=None):
    name = input_name_smi[0]
    smi = input_name_smi[1]
    out_lig_path = os.path.join(out_dir, f"{name}.sdf")

    if os.path.exists(out_lig_path):
        docked_data = parse_docked_mol(out_lig_path)
        return [docked_data[i][1] for i in range(len(docked_data))]

    input_mol = smi_gen_3d(smi)
    input_mol.SetProp("_Name", name)
    if hasattr(ref_mol, 'GetConformer'):
        input_mol = center_on_ref(input_mol, ref_mol)
    else:
        input_mol = center_on_coords(input_mol, ref_mol)
    score_list, mol_list = dock_func(rec_mol, input_mol, rec_pred_H=True, lig_pred_H=True,
                                     lig_ff_num=1, max_it=None, exhaustiveness=1,
                                     restarts=restarts,
                                     workers=1, output=out_lig_path, pose_num=pose_num,
                                     energy_range=6.0, use_gauss=True, pad_size=pad_size,
                                     gscore_params_path=gscore_params_path)
    return score_list


def run_vs(args, rec_mol, ref_mol, name_smi_list, restarts, pose_num, layer_dir,
           gscore_params_path=None):
    lig_name_score_map = {}
    workers = args.workers if args.workers else os.cpu_count()

    result_file = os.path.join(layer_dir, "runtime_state.json")
    existing_results = safe_json_load(result_file)

    todo_list = []
    for name_smi in name_smi_list:
        name = name_smi[0]
        if name in existing_results:
            lig_name_score_map[name] = existing_results[name]
        else:
            todo_list.append(name_smi)

    if len(todo_list) == 0:
        return [(ns[0], existing_results[ns[0]]) for ns in name_smi_list]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_clipdock, rec_mol, ref_mol, name_smi,
                restarts, layer_dir, pose_num, args.pad, gscore_params_path
            ): name_smi for name_smi in todo_list}

        total_futures = len(futures)
        save_interval = max(1, len(todo_list) // 100)
        for i, future in enumerate(tqdm(as_completed(futures), total=total_futures,
                                        disable=args.quiet)):
            name_smi = futures[future]
            try:
                score_data = future.result()
                best_score = min(score_data) if score_data else MAX_SCORE
                lig_name_score_map[name_smi[0]] = best_score
                existing_results[name_smi[0]] = best_score
            except Exception as e:
                clipdock_logger.error(f"Error processing {name_smi[0]}: {e}")
                lig_name_score_map[name_smi[0]] = MAX_SCORE
                existing_results[name_smi[0]] = MAX_SCORE

            if (i + 1) % save_interval == 0 or (i + 1) == total_futures:
                safe_json_save(existing_results, result_file)

    final_results = []
    for name_smi in name_smi_list:
        name = name_smi[0]
        final_results.append((name, lig_name_score_map.get(name, existing_results.get(name, MAX_SCORE))))

    return final_results


def main(args):
    setup_logging(not args.quiet)
    clipdock_logger.info(f"CLIPDock {VERSION}")
    gscore_params_path = os.path.abspath(getattr(args, 'gscore', None) or GAUSS_PARAMS_PATH)
    clipdock_logger.info(f"G-score parameters: {gscore_params_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    receptor_name = os.path.basename(args.receptor)
    work_receptor = os.path.join(args.output_dir, receptor_name)
    if os.path.abspath(args.receptor) != os.path.abspath(work_receptor):
        shutil.copy(args.receptor, work_receptor)

    if not args.skip_extract_pocket and args.ref:
        ref_name = os.path.basename(args.ref)
        work_ref = os.path.join(args.output_dir, ref_name)
        if os.path.abspath(args.ref) != os.path.abspath(work_ref):
            shutil.copy(args.ref, work_ref)

        pocket_path = os.path.splitext(work_receptor)[0] + "_pocket.pdb"
        extract_pocket(work_ref, work_receptor, pocket_path, remove_water=True)
        rec_mol = read_mol(pocket_path)
        ref_mol = read_mol(work_ref)
        if ref_mol is None:
            ref_mol = read_coords(work_ref)
    else:
        rec_mol = read_mol(work_receptor)
        if args.ref:
            ref_mol = read_mol(args.ref)
            if ref_mol is None:
                ref_mol = read_coords(args.ref)
        else:
            ref_mol = rec_mol

    name_smi_list = []
    with open(args.input_file, 'r') as read_f:
        reader = csv.reader(read_f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                name_smi_list.append((row[0], row[1]))

    clipdock_logger.info(f"Total molecules to screen: {len(name_smi_list)}")

    restarts = args.restarts
    pose_num = args.top_pose_num
    layer_name = f"pose_r{restarts}"
    layer_dir = os.path.join(args.output_dir, layer_name)
    os.makedirs(layer_dir, exist_ok=True)

    layer_results = run_vs(args, rec_mol, ref_mol, name_smi_list,
                           restarts, pose_num, layer_dir, gscore_params_path)
    layer_results.sort(key=lambda x: x[1])

    if args.top_num is not None:
        prefix_name = "final"
        generate_top_files(layer_results, args.output_dir, layer_dir, prefix_name,
                           args.top_num)

    score_by_name = dict(layer_results)
    final_results = [
        (name, score_by_name.get(name, MAX_SCORE))
        for name, _ in name_smi_list
    ]

    save_csv(final_results, args.output_dir, "results.csv", sort_results=False)
    save_csv(final_results, args.output_dir, "results_sorted.csv", sort_results=True)

    clipdock_logger.info("Screening completed successfully!")

    return final_results
