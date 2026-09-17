import numpy as np
from rdkit import Chem
from clipdock.atom_type import GscoreAtomType
from clipdock.utils.mol import protonate_mol

residue_type_map = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7,
    'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19, 'ASX': 20, 'GLX': 21,
    'CSO': 22, 'HIP': 23, 'HIE': 24, 'HID': 25, 'CYM': 26, 'CYX': 27, 'MSE': 28,
    'PTR': 29, 'TPO': 30, 'SEP': 31, 'UNK': 31
}

gscore_atom_typer = GscoreAtomType()

def mol_to_graph(mol, is_protein=False, remove_H=True):
    try:
        mol = protonate_mol(mol)
    except Exception:
        pass

    atom_types = [gscore_atom_typer.assign_atom_type_id(atom) for atom in mol.GetAtoms()]

    if remove_H:
        non_h_indices = [i for i, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() != 'H']

        atom_types = [atom_types[i] for i in non_h_indices]

        new_mol = Chem.RWMol(mol)
        for i in sorted([i for i, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() == 'H'], reverse=True):
            new_mol.RemoveAtom(i)
        mol = new_mol.GetMol()
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(num_atoms)], dtype=np.float32)

    node_features = []

    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)

        type_feat = [0.0] * 48
        type_feat[atom_types[i]] = 1.0

        if is_protein:
            res_feat = [0.0] * 32
            res_info = atom.GetPDBResidueInfo()
            if res_info:
                res_name = res_info.GetResidueName().strip().upper()
                res_feat[residue_type_map.get(res_name, 31)] = 1.0
            else:
                res_feat[31] = 1.0
            node_features.append(type_feat + res_feat)
        else:
            node_features.append(type_feat)

    edge_index, edge_attr = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bt_feat = [1.0 if bt == Chem.BondType.SINGLE else 0.0,
                   1.0 if bt == Chem.BondType.DOUBLE else 0.0,
                   1.0 if bt == Chem.BondType.TRIPLE else 0.0,
                   1.0 if bt == Chem.BondType.AROMATIC else 0.0]
        feat = bt_feat + [1.0 if bond.GetIsConjugated() else 0.0,
                          1.0 if bond.IsInRing() else 0.0,
                          float(bond.GetStereo())]


        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([feat, feat])

    edge_dim = 7

    return {
        'x': np.array(node_features, dtype=np.float32),
        'edge_index': np.array(edge_index, dtype=np.int64).T if len(edge_index) > 0 else np.zeros((2, 0), dtype=np.int64),
        'edge_attr': np.array(edge_attr, dtype=np.float32).reshape(-1, edge_dim),
        'pos': pos,
        'num_nodes': num_atoms
    }

def protein_to_graph(mol):
    return mol_to_graph(mol, is_protein=True)

def ligand_to_graph(mol):
    return mol_to_graph(mol, is_protein=False)
