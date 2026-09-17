import os
import random
from collections import deque

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

CHARGE_PATTERN = [
    (Chem.MolFromSmarts('[N;X2H1+0,X3H2+1;!a]=[C;!a]([N;X3H2+0,X4H3+1;!a])[N;X3+0,X4+1;!a]'), (0, 2, 3), (1, 0, 0), (1, 1, 1)),
    (Chem.MolFromSmarts('[N;X2H1+0,X3H2+1;!a]=[C;!a][N;X3H2+0,X4H3+1;!a]'), (0, 2), (1, 0), (1, 1)),
    (Chem.MolFromSmarts('[NX3H2;$(N-[CX4]);!a]'), (0,), (1,), (1,)),
    (Chem.MolFromSmarts('[NX3H1;$(N(-[CX4])-[CX4]);!a]'), (0,), (1,), (1,)),
    (Chem.MolFromSmarts('[NX3H0;$(N(-[CX4])(-[CX4])-[CX4]);!a]'), (0,), (1,), (1,)),
    (Chem.MolFromSmarts('[CX3](=[OX1])[O;X2H1+0,X1H0-1]'), (1, 2), (0, -1), (-1, -1)),
    (Chem.MolFromSmarts('[PX4](=[OX1])([O;X2+0,X1H0-1])([O;X2+0,X1H0-1])'), (2, 3), (-1, -1), (-1, -1)),
    (Chem.MolFromSmarts('[SX4](=[OX1])(=[OX1])([O;X2H1+0,X1H0-1])'), (3,), (-1,), (-1,)),
]


def read_mol(file_path, try_mol=True, sdf_list=False, removeHs=True, sanitize=True):
    name, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == '.pdb':
        try:
            mol = Chem.MolFromPDBFile(file_path, removeHs=removeHs, sanitize=sanitize)
        except Exception:
            mol = None
    elif ext == '.sdf':
        if sdf_list:
            try:
                return Chem.SDMolSupplier(file_path, removeHs=removeHs, sanitize=sanitize)
            except Exception:
                pass
        try:
            mol = Chem.SDMolSupplier(file_path, removeHs=removeHs, sanitize=sanitize)[0]
        except Exception:
            mol = None
        if mol is None and try_mol:
            mol2_path = name + '.mol2'
            try:
                mol = Chem.MolFromMol2File(mol2_path, removeHs=removeHs, sanitize=sanitize)
            except Exception:
                mol = None
    elif ext == '.mol2':
        try:
            mol = Chem.MolFromMol2File(file_path, removeHs=removeHs, sanitize=sanitize)
        except Exception:
            mol = None
        if mol is None and try_mol:
            sdf_path = name + '.sdf'
            try:
                mol = Chem.SDMolSupplier(sdf_path, removeHs=removeHs, sanitize=sanitize)[0]
            except Exception:
                mol = None
    else:
        mol = None
    return mol


def cal_atom_charge(atom):
    atom_symbol = atom.GetSymbol()
    if atom_symbol == 'N':
        patterns = CHARGE_PATTERN[:5]
    elif atom_symbol == 'O':
        patterns = CHARGE_PATTERN[5:]
    else:
        return 0, 0, None

    mol = atom.GetOwningMol()
    atom_idx = atom.GetIdx()
    degree_withoutH = atom.GetDegree() - sum(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())

    for p, pattern in enumerate(patterns):
        matches = mol.GetSubstructMatches(pattern[0])
        for match in matches:
            for i, pos in enumerate(pattern[1]):
                if atom_idx == match[pos]:
                    if pattern[2][i] < 0 and atom_symbol == 'O' and degree_withoutH > 1:
                        continue
                    return pattern[2][i], pattern[3][i], p
    charge = atom.GetFormalCharge()
    if charge != 0:
        return charge, charge, None
    return 0, 0, None


def protonate_mol(rdkit_mol):
    mol = Chem.RemoveAllHs(Chem.Mol(rdkit_mol))
    out_mol = Chem.Mol(mol)

    for atom in mol.GetAtoms():
        exp_charge, _, _ = cal_atom_charge(atom)
        if exp_charge != 0:
            out_atom = out_mol.GetAtomWithIdx(atom.GetIdx())
            out_atom.SetFormalCharge(exp_charge)
    Chem.SanitizeMol(out_mol)

    out_mol = Chem.AddHs(out_mol, addCoords=True)
    return out_mol


def mol_gen_3d(mol, ff="MMFF94", is_addHs=True, maxIters=10000):
    if mol is None:
        return None
    new_mol = Chem.RemoveHs(Chem.Mol(mol))
    if is_addHs:
        new_mol = protonate_mol(new_mol)

    for confId in range(new_mol.GetNumConformers()):
        new_mol.RemoveConformer(confId)

    seed = random.SystemRandom().randrange(0, 2147483647)
    ETKDG = AllChem.ETKDGv3()
    ETKDG.randomSeed = seed
    AllChem.EmbedMolecule(new_mol, ETKDG)

    if new_mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(new_mol, randomSeed=seed)
        if new_mol.GetNumConformers() == 0:
            return None

    res = -1
    if ff in ['MMFF94s', 'MMFF94'] and AllChem.MMFFGetMoleculeProperties(new_mol, mmffVariant=ff) is not None:
        try:
            res = AllChem.MMFFOptimizeMolecule(new_mol, mmffVariant=ff, maxIters=maxIters)
        except Exception:
            res = -1

    if res != 0:
        try:
            res = AllChem.UFFOptimizeMolecule(new_mol, maxIters=maxIters)
        except Exception:
            res = -1
    if res != 0:
        pass

    try:
        new_mol = center_on_ref(new_mol, mol)
    except Exception:
        pass

    return new_mol


def smi_gen_3d(smi, ff="MMFF94", is_addHs=True, maxIters=10000):
    mol = Chem.MolFromSmiles(smi)
    return mol_gen_3d(mol, ff, is_addHs, maxIters)


def center_on_coords(mol, ref_coords):
    mol = Chem.Mol(mol)
    lig_coords = mol.GetConformer().GetPositions()
    shift = ref_coords.mean(axis=0) - lig_coords.mean(axis=0)

    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (pos.x + shift[0], pos.y + shift[1], pos.z + shift[2]))
    return mol


def center_on_ref(mol, ref_mol):
    ref_coords = ref_mol.GetConformer().GetPositions()
    return center_on_coords(mol, ref_coords)


def calc_rmsd(mol1, mol2, maxMatches=10000):
    try:
        rmsd = AllChem.CalcRMS(Chem.RemoveAllHs(Chem.Mol(mol1)),
                               Chem.RemoveAllHs(Chem.Mol(mol2)),
                               maxMatches=maxMatches)
    except Exception:
        rmsd = 99
    return rmsd


def mols_cluster(mols, threshold=0.5):
    morgan_generator = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprints = [morgan_generator.GetFingerprint(mol) for mol in mols]
    dists = []
    nfps = len(fingerprints)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        dists.extend([1 - x for x in sims])

    clusters = Butina.ClusterData(dists, nfps, threshold, isDistData=True)
    return clusters


def get_intra_mask(mol, max_depth=None, include_dist=False):
    coords = np.ascontiguousarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    lig_size = len(coords)
    mask = np.ascontiguousarray(np.full((lig_size, lig_size), 100000000, dtype=np.int32))
    dist = np.ascontiguousarray(np.zeros((lig_size, lig_size), dtype=np.float64)) if include_dist else None

    for i in range(lig_size):
        mask[i][i] = 1
        visited = {i: 1}
        queue = deque([i])

        while queue:
            current_idx = queue.popleft()
            current_deep = visited[current_idx]

            if max_depth is not None and current_deep >= max_depth:
                continue

            current_atom = mol.GetAtomWithIdx(current_idx)
            for neighbor in current_atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in visited:
                    continue
                new_deep = current_deep + 1
                visited[neighbor_idx] = new_deep
                queue.append(neighbor_idx)

                mask[i][neighbor_idx] = new_deep
                mask[neighbor_idx][i] = new_deep
                if include_dist:
                    pos1 = coords[i]
                    pos2 = coords[neighbor_idx]
                    new_dist = np.linalg.norm(pos1 - pos2)
                    dist[i][neighbor_idx] = new_dist
                    dist[neighbor_idx][i] = new_dist
    return mask, dist


def parse_docked_mol(docked_path, ref_mol=None):
    docked_mol_list = read_mol(docked_path, sdf_list=True)

    lig_mol_data = []
    for i in range(len(docked_mol_list)):
        mol = docked_mol_list[i]
        if any(calc_rmsd(pair[0], mol) < 1.0 for pair in lig_mol_data):
            continue
        try:
            score = float(mol.GetProp('CLIPDock score'))
        except Exception:
            try:
                score = float(mol.GetProp('Inter score'))
            except Exception:
                score = 0.0
        if ref_mol is not None:
            try:
                rmsd = calc_rmsd(ref_mol, mol)
            except Exception:
                rmsd = 99
            lig_mol_data.append((mol, score, rmsd))
        else:
            lig_mol_data.append((mol, score, 999))
    lig_mol_data.sort(key=lambda x: x[1])
    return lig_mol_data
