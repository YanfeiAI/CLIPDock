"""Cross-version regression driver.

Executed *inside* the process of one CLIPDock tree (v1.0.0, main or dev) by
``tests/regression_support.py``.  It only uses APIs that already exist in
v1.0.0, emits float64-exact ``.npz`` artifacts plus a JSON manifest, and never
writes outside ``--out``.

    PYTHONPATH=<tree> python tests/regression_driver.py \
        --molecules molecules.json --task t1 t2 --out <dir>
"""

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np

def _meta(extra=None):
    import clipdock
    import numba
    import rdkit
    import scipy
    meta = {
        'clipdock_file': os.path.abspath(clipdock.__file__),
        'clipdock_version': getattr(clipdock, '__version__', 'unknown'),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'numba': numba.__version__,
        'rdkit': rdkit.__version__,
    }
    try:
        import torch
        meta['torch'] = torch.__version__
        meta['cuda'] = torch.cuda.is_available()
    except Exception:
        meta['torch'] = None
    meta['inputs'] = {}
    for entry in _load_entries(os.environ['CLIPDOCK_MOLECULES_JSON']):
        for key in ('ligand', 'pocket'):
            path = entry.get(key)
            if path and os.path.isfile(path):
                with open(path, 'rb') as f:
                    meta['inputs'][f'{entry["name"]}:{key}'] = hashlib.sha256(f.read()).hexdigest()[:16]
    meta['env'] = {k: v for k, v in os.environ.items()
                   if k.startswith(('NUMBA', 'OMP_', 'MKL_', 'OPENBLAS', 'CLIPDOCK'))}
    meta['executable'] = sys.executable
    if extra:
        meta.update(extra)
    return meta


def _load_entries(path):
    with open(path) as f:
        data = json.load(f)
    return data['entries']


def _pocket(path):
    from clipdock.utils.mol import read_mol
    from clipdock.utils.pocket import remove_clashing_bonds
    mol = read_mol(path)
    if mol is None:
        mol = remove_clashing_bonds(path)
    return mol


def _hex(a):
    return np.asarray(a, dtype=np.float64).ravel().view(np.uint64).astype(str).tolist()


def _protonated(lig_mol):
    from rdkit import Chem
    from clipdock.utils.mol import protonate_mol
    return protonate_mol(Chem.RemoveHs(Chem.Mol(lig_mol)))


def _optimizer(rec_mol, lig_mol, seed=1234, max_iter=48):
    from clipdock.engine import PoseOptimizer
    from clipdock.utils.constant import DOCKING_PARAMS_PATH, GAUSS_PARAMS_PATH
    return PoseOptimizer(rec_mol, lig_mol, DOCKING_PARAMS_PATH, 1, seed, max_iter,
                         dist_cutoff=8.0, pad_size=8.0,
                         gscore_params_path=GAUSS_PARAMS_PATH)


# --------------------------------------------------------------------- tasks
def task_t0(entries):
    """Extract pockets for entries that only ship a full receptor."""
    from clipdock.utils.pocket import extract_pocket
    mapping = {}
    for e in entries:
        if 'pocket' in e or 'receptor' not in e:
            continue
        out = os.path.join(OUT_DIR, f'{e["name"]}_pocket.pdb')
        if not os.path.isfile(out):
            extract_pocket(e['ligand'], e['receptor'], out, remove_water=True)
        if os.path.isfile(out):
            mapping[e['name']] = out
    return {}, mapping


def task_t1(entries):
    """Per-atom (exp_charge, charge, charge_part) of every ligand and pocket."""
    from clipdock.utils.mol import cal_atom_charge, protonate_mol
    try:
        from clipdock.utils.mol import calc_atom_charges
    except ImportError:  # v1.0.0 / main have no vectorised helper
        calc_atom_charges = None
    from rdkit import Chem
    arrays, batch_ok = {}, {}
    for e in entries:
        for kind, mol in (('lig', e['lig_mol']), ('poc', e['poc_mol'])):
            for tag, m in (('raw', mol), ('prot', protonate_mol(Chem.RemoveHs(Chem.Mol(mol))))):
                tri = []
                for atom in m.GetAtoms():
                    exp, chg, part = cal_atom_charge(atom)
                    tri.append((int(exp), int(chg), -99 if part is None else int(part)))
                arrays[f'{e["name"]}:{kind}:{tag}'] = np.asarray(tri, dtype=np.int64).reshape(-1, 3)
        # candidate-only: the vectorised helper must agree with the per-atom one
        if calc_atom_charges is None:
            continue
        for kind, mol in (('lig', e['lig_mol']), ('poc', e['poc_mol'])):
            exp, chg, part = calc_atom_charges(mol)
            tri = np.asarray([(int(a), int(b), -99 if c is None else int(c))
                              for a, b, c in zip(exp, chg, part)], dtype=np.int64).reshape(-1, 3)
            arrays[f'{e["name"]}:{kind}:batch'] = tri
            ref = arrays[f'{e["name"]}:{kind}:raw']
            batch_ok[f'{e["name"]}:{kind}'] = bool(np.array_equal(ref, tri))
    return arrays, {'batch_matches_per_atom': batch_ok}


def task_t2(entries):
    """Atom type ids for both type classes and both keep_H settings."""
    from clipdock.atom_type import EscoreAtomType, GscoreAtomType
    arrays, dtypes = {}, {}
    classes = (('E', EscoreAtomType()), ('G', GscoreAtomType()))
    for e in entries:
        for kind, mol in (('lig', e['lig_mol']), ('poc', e['poc_mol'])):
            for cname, tc in classes:
                for keep_h in (True, False):
                    ids = tc.assign_atom_types(mol, keep_H=keep_h)
                    key = f'{e["name"]}:{kind}:{cname}:{int(keep_h)}'
                    arrays[key] = np.asarray(ids)
                    dtypes[key] = str(np.asarray(ids).dtype)
    return arrays, {'dtypes': dtypes}


def task_t3(entries):
    """Protonation: atom count, charges, bonds, coordinates, SMILES."""
    from rdkit import Chem
    from clipdock.utils.mol import protonate_mol
    arrays, info = {}, {}
    for e in entries:
        for kind, mol in (('lig', e['lig_mol']), ('poc', e['poc_mol'])):
            try:
                prot = protonate_mol(Chem.RemoveHs(Chem.Mol(mol)))
            except Exception as exc:  # keep the failure visible instead of skipping
                info[f'{e["name"]}:{kind}'] = {'error': repr(exc)}
                continue
            key = f'{e["name"]}:{kind}'
            bonds = np.asarray([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble() * 100))
                                for b in prot.GetBonds()], dtype=np.int64).reshape(-1, 3)
            arrays[key + ':charges'] = np.asarray([a.GetFormalCharge() for a in prot.GetAtoms()], dtype=np.int64)
            arrays[key + ':bonds'] = bonds
            arrays[key + ':coords'] = np.ascontiguousarray(prot.GetConformer().GetPositions(), dtype=np.float64)
            info[key] = {
                'n_atoms': prot.GetNumAtoms(),
                'n_conformers': prot.GetNumConformers(),
                'smiles': Chem.MolToSmiles(prot),
                'n_bonds': int(bonds.shape[0]),
            }
    return arrays, {'molecules': info}


def _poses(opt, n, seed=1234):
    """Deterministic pose parameters from the optimizer's own RNG stream."""
    opt.rng = __import__('numpy').random.Generator(
        __import__('numpy').random.PCG64(seed))
    return [opt.random_init_x() for _ in range(n)]


def task_t4(entries):
    """E-score, intra score, G-score and their gradients on fixed poses."""
    from clipdock.scoring import EScorer, GScorer
    from clipdock.utils.constant import DOCKING_PARAMS_PATH, GAUSS_PARAMS_PATH
    from clipdock.utils.trans import apply_transformations
    arrays = {}
    for e in entries:
        lig_h = _protonated(e['lig_mol'])
        rec_h = _protonated(e['poc_mol'])
        opt = _optimizer(rec_h, lig_h)
        esc = EScorer(rec_h, lig_h, DOCKING_PARAMS_PATH, 8.0, 8.0)
        gsc = GScorer(rec_h, lig_h, GAUSS_PARAMS_PATH, 8.0, 8.0)
        xs = _poses(opt, 12)
        scores, intra, gscores, grads = [], [], [], []
        for x in xs:
            coords = apply_transformations(x, opt.ori_coords, opt.anchor, opt.rot_atom_pairs,
                                           opt.rot_affected_indices, opt.rot_affected_offsets)
            s, grad = esc.calc_escore_with_grad(coords)
            scores.append([esc.calc_escore(coords), s])
            intra.append(esc.calc_score_intra(esc.lig_atom_types, coords, None, esc.uff_e_min,
                                              esc.uff_r_min, esc.r_d, 8.0, esc.lig_intra_mask))
            grads.append(np.array(grad, copy=True))
            gscores.append(gsc.calc_gscore(opt.pose_with_hydrogens(coords).GetConformer().GetPositions()
                                          if hasattr(opt, 'pose_with_hydrogens')
                                          else _h_coords(opt, coords)))
        arrays[e['name'] + ':scores'] = np.asarray(scores, dtype=np.float64)
        arrays[e['name'] + ':intra'] = np.asarray(intra, dtype=np.float64)
        arrays[e['name'] + ':gscore'] = np.asarray(gscores, dtype=np.float64)
        arrays[e['name'] + ':atom_grad'] = np.asarray(grads, dtype=np.float64)
        arrays[e['name'] + ':x'] = np.asarray(xs, dtype=np.float64)
    return arrays, {}


def _h_coords(opt, coords):
    """v1.0.0 protonation path used by gauss_inter/minimize_pose."""
    from rdkit import Chem
    from clipdock.utils.mol import protonate_mol
    mol = Chem.Mol(opt.lig_mol)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, coords[i])
    return protonate_mol(mol).GetConformer().GetPositions()


def task_t5(entries):
    """Parameter gradient of the L-BFGS-B objective on the same poses."""
    arrays = {}
    for e in entries:
        lig_h = _protonated(e['lig_mol'])
        rec_h = _protonated(e['poc_mol'])
        opt = _optimizer(rec_h, lig_h)
        xs = _poses(opt, 12)
        f, g = [], []
        for x in xs:
            val, grad = opt.objective_with_grad(x)
            f.append(val)
            g.append(np.array(grad, copy=True))
        arrays[e['name'] + ':f'] = np.asarray(f, dtype=np.float64)
        arrays[e['name'] + ':g'] = np.asarray(g, dtype=np.float64)
    return arrays, {}


def task_t6(entries):
    """Rotatable-bond decomposition, transformed coordinates and projection."""
    from clipdock.utils.rot import find_rotatable_bonds
    from clipdock.utils.trans import (apply_transformations, apply_transformations_inplace,
                                      project_transform_gradients_inplace)
    arrays = {}
    for e in entries:
        lig_h = _protonated(e['lig_mol'])
        rec_h = _protonated(e['poc_mol'])
        from rdkit import Chem
        lig_woh = Chem.RemoveAllHs(Chem.Mol(lig_h))
        pairs, idx, offs = find_rotatable_bonds(lig_woh)
        arrays[e['name'] + ':pairs'] = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        arrays[e['name'] + ':indices'] = np.asarray(idx, dtype=np.int64)
        arrays[e['name'] + ':offsets'] = np.asarray(offs, dtype=np.int64)
        opt = _optimizer(rec_h, lig_h)
        xs = _poses(opt, 6)
        coords_out, param_out = [], []
        for x in xs:
            coords_out.append(apply_transformations(x, opt.ori_coords, opt.anchor,
                                                    opt.rot_atom_pairs, opt.rot_affected_indices,
                                                    opt.rot_affected_offsets))
            atom_grad = np.ascontiguousarray(np.sin(np.arange(opt.ori_coords.size)
                                                    .reshape(opt.ori_coords.shape)).astype(np.float64))
            buf = np.empty_like(opt.ori_coords)
            ibuf = np.empty_like(opt.ori_coords)
            gbuf = np.zeros(6 + opt.rot_num, dtype=np.float64)
            rot = np.empty((3, 3), dtype=np.float64)
            apply_transformations_inplace(x, opt.ori_coords, opt.anchor, opt.rot_atom_pairs,
                                          opt.rot_affected_indices, opt.rot_affected_offsets,
                                          buf, rot)
            project_transform_gradients_inplace(x, opt.ori_coords, opt.anchor, opt.rot_atom_pairs,
                                               opt.rot_affected_indices, opt.rot_affected_offsets,
                                               atom_grad, gbuf, ibuf, ibuf.copy(), rot)
            param_out.append(np.array(gbuf, copy=True))
        arrays[e['name'] + ':coords'] = np.asarray(coords_out, dtype=np.float64)
        arrays[e['name'] + ':pgrad'] = np.asarray(param_out, dtype=np.float64)
    return arrays, {}


def task_t7(entries):
    """RMSD, deduplication and pose serialisation."""
    from rdkit import Chem
    from clipdock.dock import filter_mol_list, process_pose_list
    from clipdock.utils.mol import calc_rmsd
    arrays, info = {}, {}
    for e in entries:
        lig_h = _protonated(e['lig_mol'])
        rec_h = _protonated(e['poc_mol'])
        opt = _optimizer(rec_h, lig_h)
        xs = _poses(opt, 24)
        from clipdock.utils.trans import apply_transformations
        pose_list = []
        for i, x in enumerate(xs):
            coords = apply_transformations(x, opt.ori_coords, opt.anchor, opt.rot_atom_pairs,
                                           opt.rot_affected_indices, opt.rot_affected_offsets)
            pose_list.append([float(i) * 0.01, coords])
        mols = process_pose_list(lig_h, pose_list)
        arrays[e['name'] + ':rmsd'] = np.asarray(
            [[calc_rmsd(mols[i][1], mols[j][1], maxMatches=100) for j in range(len(mols))]
             for i in range(len(mols))], dtype=np.float64)
        score_list = [m[0] for m in mols]
        mol_list = [m[1] for m in mols]
        kept_s, kept_m = filter_mol_list(score_list, mol_list, 6.0, 1.0, 9)
        arrays[e['name'] + ':kept_scores'] = np.asarray(kept_s, dtype=np.float64)
        info[e['name']] = {
            'n_kept': len(kept_s),
            'molblocks': [hashlib.sha1(Chem.MolToMolBlock(m).encode()).hexdigest() for m in kept_m],
            'atom_counts': [m.GetNumAtoms() for m in kept_m],
        }
    return arrays, {'poses': info}


def task_t8(entries):
    """Scorer reuse: repeated construction must not leak state across ligands."""
    from clipdock.scoring import EScorer, GScorer
    from clipdock.utils.constant import DOCKING_PARAMS_PATH, GAUSS_PARAMS_PATH
    from clipdock.utils.trans import apply_transformations
    arrays = {}
    for e in entries:
        lig_h = _protonated(e['lig_mol'])
        rec_h = _protonated(e['poc_mol'])
        opt = _optimizer(rec_h, lig_h)
        xs = _poses(opt, 4)
        coords = [apply_transformations(x, opt.ori_coords, opt.anchor, opt.rot_atom_pairs,
                                        opt.rot_affected_indices, opt.rot_affected_offsets)
                  for x in xs]
        runs = []
        for round_idx in range(3):
            esc = EScorer(rec_h, lig_h, DOCKING_PARAMS_PATH, 8.0, 8.0)
            gsc = GScorer(rec_h, lig_h, GAUSS_PARAMS_PATH, 8.0, 8.0)
            runs.append([esc.calc_escore(c) for c in coords]
                        + [gsc.calc_gscore(opt.pose_with_hydrogens(c).GetConformer().GetPositions()
                                           if hasattr(opt, 'pose_with_hydrogens')
                                           else _h_coords(opt, c)) for c in coords])
        arrays[e['name'] + ':repeat'] = np.asarray(runs, dtype=np.float64)
        origin, dims, starts, counts, cell_types, cell_coords = EScorer.build_static_cell_list(
            esc.rec_atom_types, esc.rec_coords, 8.0)
        arrays[e['name'] + ':cell_origin'] = np.asarray(origin, dtype=np.float64)
        arrays[e['name'] + ':cell_dims'] = np.asarray(dims, dtype=np.int64)
        arrays[e['name'] + ':cell_starts'] = np.asarray(starts, dtype=np.int64)
        arrays[e['name'] + ':cell_counts'] = np.asarray(counts, dtype=np.int64)
        arrays[e['name'] + ':cell_types'] = np.asarray(cell_types, dtype=np.int64)
        arrays[e['name'] + ':cell_coords'] = np.asarray(cell_coords, dtype=np.float64)
    return arrays, {}


def task_t9(entries):
    """Seeded search trajectory with the conformer generator frozen."""
    from rdkit import Chem
    import clipdock.engine as engine_mod
    from clipdock.utils.rot import find_rotatable_bonds
    engine_mod.mol_gen_3d = lambda mol, *a, **kw: Chem.Mol(mol)
    arrays = {}
    for e in entries:
        lig_h = _protonated(e['lig_mol'])
        rec_h = _protonated(e['poc_mol'])
        from clipdock.dock import cal_mol_iter
        max_it = cal_mol_iter(lig_h, e=1)
        scores, coords = [], []
        for r in range(2):
            opt = _optimizer(rec_h, lig_h, seed=777 + r, max_iter=max_it)
            poses = opt.evolve(pose_num=9, use_gauss=True)
            scores.append([p[0] for p in poses])
            coords.append([np.asarray(p[1], dtype=np.float64) for p in poses])
        arrays[e['name'] + ':scores'] = np.asarray(scores, dtype=np.float64)
        arrays[e['name'] + ':coords'] = np.asarray(coords, dtype=np.float64)
        arrays[e['name'] + ':max_it'] = np.asarray([max_it], dtype=np.int64)
        arrays[e['name'] + ':rot'] = np.asarray(
            [len(find_rotatable_bonds(Chem.RemoveAllHs(Chem.Mol(lig_h)))[0])], dtype=np.int64)
    return arrays, {}


def task_t10(entries):
    """Deterministic full-pipeline docking: minimize only (no random search)."""
    from rdkit import Chem
    from clipdock.dock import dock_func, score_only
    arrays, info = {}, {}
    for e in entries:
        rec_h = _protonated(e['poc_mol'])
        try:
            scores, mols = dock_func(rec_h, e['lig_mol'], rec_pred_H=True, lig_pred_H=True,
                                     lig_ff_num=1, restarts=1, workers=1, output=None,
                                     pose_num=1, minimize=True)
            arrays[e['name'] + ':min_score'] = np.asarray(scores, dtype=np.float64)
            arrays[e['name'] + ':min_coords'] = np.ascontiguousarray(
                mols[0].GetConformer().GetPositions(), dtype=np.float64)
            info[e['name']] = {'block': hashlib.sha1(Chem.MolToMolBlock(mols[0]).encode()).hexdigest()}
        except Exception as exc:
            info[e['name']] = {'error': repr(exc)}
    try:
        rec_h = _protonated(entries[0]['poc_mol'])
        arrays['score_only'] = np.asarray(
            score_only(rec_h, [e['lig_mol'] for e in entries], lig_pred_H=False),
            dtype=np.float64)
    except Exception as exc:
        info['score_only'] = {'error': repr(exc)}
    return arrays, {'docking': info}


def task_t11(entries):
    """VS-score graph conversion and CPU inference on frozen poses."""
    import torch
    from rdkit import Chem
    from model.graph import ligand_to_graph, protein_to_graph
    from model.graph_utils import collate_graph_attributes
    from model.train_vs import VSTrainer
    import model.graph_score as gs

    shared = os.environ.get('CLIPDOCK_VS_CKPT')
    if not shared or not os.path.isfile(shared):
        return {}, {'error': f'checkpoint missing: {shared}'}
    rec_mol = Chem.MolFromPDBFile(entries[0]['pocket'])
    if rec_mol is None:
        return {}, {'error': 'pocket unreadable'}
    g_rec = protein_to_graph(rec_mol)
    arrays = {'prot:x': np.asarray(g_rec['x']), 'prot:edge_index': np.asarray(g_rec['edge_index']),
              'prot:edge_attr': np.asarray(g_rec['edge_attr']), 'prot:pos': np.asarray(g_rec['pos'])}
    lig_graphs = []
    for e in entries:
        suppl = Chem.SDMolSupplier(e['ligand'], removeHs=False)
        mol = suppl[0] if len(suppl) else None
        if mol is None:
            continue
        g = ligand_to_graph(mol)
        if g is None:
            continue
        lig_graphs.append(g)
        arrays[f'lig:{e["name"]}:x'] = np.asarray(g['x'])
        arrays[f'lig:{e["name"]}:edge_index'] = np.asarray(g['edge_index'])
        arrays[f'lig:{e["name"]}:edge_attr'] = np.asarray(g['edge_attr'])
        arrays[f'lig:{e["name"]}:pos'] = np.asarray(g['pos'])

    device = torch.device('cpu')
    model = VSTrainer.load_from_checkpoint(shared, map_location=device)
    model.eval()
    scores = []
    bs = 12
    for start in range(0, len(lig_graphs), bs):
        chunk = lig_graphs[start:start + bs]
        data_list = [{'prot': g_rec, 'lig': g} for g in chunk]
        batch = (collate_graph_attributes(data_list, 'prot'),
                 collate_graph_attributes(data_list, 'lig'))
        scores.extend(gs.compute_mdn_scores_batch(model, batch, device))
    arrays['scores'] = np.asarray([s if s is not None else np.nan for s in scores], dtype=np.float64)
    return arrays, {'n_ligands': len(lig_graphs)}


OUT_DIR = None


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--molecules', required=True)
    ap.add_argument('--task', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    OUT_DIR = args.out
    entries = _load_entries(args.molecules)

    # gen_lig_ff_coords() re-embeds with a system-random ETKDG seed; freeze it so
    # every version starts each task from the same internal geometry.
    from rdkit import Chem
    import clipdock.engine as engine_mod
    engine_mod.mol_gen_3d = lambda mol, *a, **kw: Chem.Mol(mol)

    os.environ['CLIPDOCK_MOLECULES_JSON'] = os.path.abspath(args.molecules)
    meta = _meta({'tasks': args.task, 'n_entries': len(entries)})
    if 't0' in args.task:
        arrays, mapping = task_t0(entries)
        np.savez_compressed(os.path.join(args.out, 't0.npz'), **arrays)
        with open(os.path.join(args.out, 't0.json'), 'w') as f:
            json.dump(mapping, f, indent=1, sort_keys=True)
        for e in entries:
            e['pocket'] = mapping.get(e['name'], e.get('pocket'))
        print(f't0: {len(mapping)} pockets')

    if any(task != 't0' for task in args.task):
        from clipdock.utils.mol import read_mol
        for e in entries:
            if 'pocket' not in e:
                raise SystemExit(f'missing pocket for {e["name"]}')
            e['lig_mol'] = read_mol(e['ligand'], removeHs=False)
            e['poc_mol'] = _pocket(e['pocket'])
            if e['lig_mol'] is None or e['poc_mol'] is None:
                raise SystemExit(f'cannot read inputs for {e["name"]}')
    for task in args.task:
        if task == 't0':
            continue
        func = globals().get(f'task_{task}')
        if func is None:
            continue
        arrays, info = func(entries)
        np.savez_compressed(os.path.join(args.out, f'{task}.npz'), **arrays)
        with open(os.path.join(args.out, f'{task}.json'), 'w') as f:
            json.dump(info, f, indent=1, sort_keys=True)
        print(f'{task}: {len(arrays)} arrays')
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
