import logging
import random

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
import sys
import os

from clipdock.engine import PoseOptimizer

from clipdock.engine import score_only
from clipdock.utils.calc import batch_process
from clipdock.utils.constant import DOCKING_PARAMS_PATH, GAUSS_PARAMS_PATH, VERSION
from clipdock.utils.mol import protonate_mol, calc_rmsd, center_on_ref, read_mol, mol_gen_3d
from clipdock.utils.pocket import extract_pocket, remove_clashing_bonds
from clipdock.utils.timing import format_timing, take_timing_snapshot, timing_delta
from clipdock.visualize import analyse_scorer_atom_contrib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
clipdock_logger = logging.getLogger('clipdock')
DEFAULT_WORKERS = 8

rdkit_logger = logging.getLogger('rdkit')
rdkit_logger.disabled = True
RDLogger.DisableLog('rdApp.warning')


def setup_logging(verbose=True):
    level = logging.INFO if verbose else logging.WARNING

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    clipdock_logger.setLevel(level)


def save_docked_mol(output, score_list, mol_list):
    output_parent = os.path.dirname(os.path.abspath(os.fspath(output)))
    os.makedirs(output_parent, exist_ok=True)
    with Chem.SDWriter(output) as writer:
        for i, mol in enumerate(mol_list):
            mol.SetProp("CLIPDock score", "%6.3lf" % (score_list[i]))
            writer.write(mol)


def process_dock(optimizer, minimize=False, pose_num=1, refresh_lig=False,
                 use_gauss=False):
    if refresh_lig:
        optimizer.refresh_lig()
    if minimize:
        return optimizer.minimize_pose()
    return optimizer.evolve(pose_num=pose_num, use_gauss=use_gauss)


def cal_mol_iter(mol, e=1):
    prop_mol = Chem.RemoveHs(Chem.Mol(mol))
    Chem.SanitizeMol(prop_mol)
    float_it = 8 * (6 + Descriptors.NumRotatableBonds(Chem.RemoveHs(Chem.Mol(prop_mol))))
    mid_iter = 128
    if float_it > mid_iter:
        float_it = mid_iter + (float_it - mid_iter) / ((float_it - mid_iter) / mid_iter + 1)
    max_iter = int(np.round(e * float_it))
    return max_iter


def filter_mol_list(score_list, mol_list, energy_range, min_rmsd, pose_num):
    new_score_list = []
    new_mol_list = []
    for i in range(len(mol_list)):
        if score_list[i] - score_list[0] >= energy_range:
            break
        if any(calc_rmsd(pmol, mol_list[i], maxMatches=100) < min_rmsd for pmol in new_mol_list):
            continue
        new_score_list.append(score_list[i])
        new_mol_list.append(mol_list[i])
        if len(new_mol_list) >= pose_num:
            break
    return new_score_list, new_mol_list


def process_pose_list(lig_mol, pose_list):
    result = []
    for p in pose_list:
        output_mol = Chem.RemoveAllHs(Chem.Mol(lig_mol))
        conf = output_mol.GetConformer()
        for i in range(output_mol.GetNumAtoms()):
            conf.SetAtomPosition(i, p[1][i])

        result.append([p[0], output_mol])
    return result


def dock_func(rec_mol, lig_mol, rec_pred_H=True, lig_pred_H=True,
              lig_ff_num=1, max_it=None, exhaustiveness=1, restarts=32,
              workers=1, output=None, pose_num=9,
              min_rmsd=1.0, energy_range=6.0,
              minimize=False, visualize=False,
              use_gauss=True, dist_cutoff=8.0, pad_size=8.0,
              gscore_params_path=GAUSS_PARAMS_PATH):
    gscore_params_path = gscore_params_path or GAUSS_PARAMS_PATH
    if output is not None:
        output_parent = os.path.dirname(os.path.abspath(os.fspath(output)))
        os.makedirs(output_parent, exist_ok=True)
    if rec_pred_H:
        rec_mol = protonate_mol(Chem.RemoveHs(Chem.Mol(rec_mol)))
    if lig_pred_H:
        lig_mol = protonate_mol(Chem.RemoveHs(Chem.Mol(lig_mol)))

    if max_it is None:
        max_it = cal_mol_iter(lig_mol, e=exhaustiveness)

    seed = random.SystemRandom().randrange(0, 2147483647)
    if minimize:
        lig_ff_num = 0
        pose_num = 1
        restarts = 1
        workers = 1
    docked_mol = []
    opter = PoseOptimizer(rec_mol, lig_mol, DOCKING_PARAMS_PATH, lig_ff_num, seed, max_it,
                          dist_cutoff=dist_cutoff, pad_size=pad_size,
                          gscore_params_path=gscore_params_path)

    if pose_num == 1:
        return_pose_num = 1
    else:
        return_pose_num = int(1e8)
    workers = min(workers, restarts)

    args_list = [(opter, minimize, return_pose_num, idx > 0, use_gauss)
                 for idx in range(restarts)]
    result = batch_process(func=process_dock, args_list=args_list, workers=workers, use_tqdm=False)
    for pose_list in result:
        docked_mol.extend(process_pose_list(lig_mol, pose_list))

    docked_mol.sort(key=lambda x: x[0])
    score_list = [docked_mol[i][0] for i in range(len(docked_mol))]
    mol_list = [docked_mol[i][1] for i in range(len(docked_mol))]
    score_list, mol_list = filter_mol_list(score_list, mol_list, energy_range, min_rmsd, pose_num)

    if output is not None and visualize:
        analyse_scorer_atom_contrib(
            mol_list[0], rec_mol, output, gscore_params_path=gscore_params_path
        )

    if output is not None:
        save_docked_mol(output, score_list, mol_list)

    return score_list, mol_list


def score_input_file(args):
    """Score every input record in place and preserve ligand data in the output."""
    if os.path.realpath(args.ligand) == os.path.realpath(args.output):
        raise ValueError("--score_only requires a new output SDF, different from the input file.")
    if os.path.splitext(args.ligand)[1].lower() == '.sdf':
        try:
            mol_list = list(Chem.SDMolSupplier(args.ligand, removeHs=False))
        except OSError as exc:
            raise ValueError(f"Cannot read input SDF: {args.ligand}") from exc
    else:
        mol_list = [read_mol(args.ligand, removeHs=False)]
    if not mol_list:
        raise ValueError("No molecules found in the input ligand file.")
    for idx, mol in enumerate(mol_list, 1):
        if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
            raise ValueError(f"Cannot score ligand record {idx}: unreadable molecule or missing coordinates.")

    rec_mol = read_mol(args.receptor)
    if rec_mol is None:
        rec_mol = remove_clashing_bonds(args.receptor)
    if args.ref:
        clipdock_logger.info("--score_only uses input ligand coordinates; --ref is ignored.")
    scores = score_only(rec_mol, mol_list, lig_pred_H=False,
                        gscore_params_path=getattr(args, 'gscore', None))
    save_docked_mol(args.output, scores, mol_list)
    clipdock_logger.info("%8s, %s, %s", "record", "name", "CLIPDock score")
    for idx, (mol, score) in enumerate(zip(mol_list, scores), 1):
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''
        clipdock_logger.info("%8d, %s, %.3f", idx, name, score)
    if args.visualize:
        for idx, mol in enumerate(mol_list, 1):
            base, _ = os.path.splitext(args.output)
            analyse_scorer_atom_contrib(Chem.Mol(mol), rec_mol, f"{base}_record{idx}.sdf",
                                       lig_pred_H=False,
                                       gscore_params_path=getattr(args, 'gscore', None))
    clipdock_logger.info(f"Saving results to {args.output}")
    return scores


def main(args):
    setup_logging(not args.quiet)
    clipdock_logger.info(f"CLIPDock {VERSION}")
    timing_start = take_timing_snapshot()

    if args.score_only:
        scores = score_input_file(args)
        wall_seconds, cpu_seconds = timing_delta(timing_start)
        clipdock_logger.info(f"{format_timing(wall_seconds, cpu_seconds)}, Workers: 1")
        return min(scores)

    receptor = args.receptor
    if not args.skip_extract_pocket:
        pocket_path = os.path.splitext(receptor)[0] + "_pocket.pdb"
        if args.ref is not None:
            extract_pocket(args.ref, receptor, pocket_path, remove_water=True)
            receptor = pocket_path
            clipdock_logger.info(f"Pre-extracted pocket saved to: {pocket_path}")
        elif args.ligand is not None:
            extract_pocket(args.ligand, receptor, pocket_path, remove_water=True)
            receptor = pocket_path
            clipdock_logger.info(f"Pre-extracted pocket saved to: {pocket_path}")

    rec_mol = read_mol(receptor)
    if rec_mol is None:
        rec_mol = remove_clashing_bonds(receptor)
    clipdock_logger.info(f"Loading pocket: {receptor}, {rec_mol.GetNumAtoms()} atoms")

    if args.ligand is None:
        lig_info = args.smi
        lig_mol = Chem.MolFromSmiles(args.smi)
        if lig_mol is None:
            raise ValueError("Failed to read SMILES.")
        lig_mol = mol_gen_3d(lig_mol)
        if lig_mol is None:
            raise ValueError("Failed to generate 3D structure from SMILES.")
        lig_mol.SetProp("_Name", os.path.basename(args.output).split('.')[0])
    else:
        lig_info = args.ligand
        lig_mol = read_mol(args.ligand)
        if lig_mol is None:
            raise ValueError("Failed to read Ligand.")
        lig_mol.SetProp("_Name", os.path.basename(args.output).split('.')[0])

    if args.ref is not None:
        ref_mol = read_mol(args.ref)
    elif args.ligand is not None:
        ref_mol = read_mol(args.ligand)
    else:
        ref_mol = Chem.Mol(rec_mol)
    lig_mol = center_on_ref(lig_mol, ref_mol)
    clipdock_logger.info(f"Loading ligand: {lig_info}, {lig_mol.GetNumAtoms()} atoms")

    if args.workers is None:
        requested_workers = DEFAULT_WORKERS
    else:
        requested_workers = max(args.workers, 1)
    workers_used = min(requested_workers, max(args.restarts, 1))
    if args.minimize:
        workers_used = 1
    score_list, _ = dock_func(rec_mol, lig_mol, rec_pred_H=True, lig_pred_H=True,
                             lig_ff_num=1, pose_num=args.pose_num,
                             max_it=None, exhaustiveness=1,
                             restarts=args.restarts, workers=requested_workers,
                             output=args.output,
                             minimize=args.minimize,
                             visualize=args.visualize,
                             pad_size=args.pad,
                             gscore_params_path=getattr(args, 'gscore', None))
    wall_seconds, cpu_seconds = timing_delta(timing_start)
    clipdock_logger.info("%8s, %8s" % (
        "rank", "CLIPDock score"))
    for idx, score in enumerate(score_list):
        formatted_scores = tuple([idx + 1] + [score])
        clipdock_logger.info("%8d, %8.3f" % formatted_scores)
    clipdock_logger.info(f"Best CLIPDock score: {score_list[0]:.3f}")
    clipdock_logger.info(
        f"{format_timing(wall_seconds, cpu_seconds)}, Workers: {workers_used}"
    )
    clipdock_logger.info(f"Saving results to {args.output}")
    return score_list[0]
