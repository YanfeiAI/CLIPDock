import numpy as np
from rdkit import Chem


def get_degree(atom):
    return sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() != 'H')


def is_rotatable_bond(bond):
    if bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    if bond.IsInRing():
        return False
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    if get_degree(a1) <= 1 or get_degree(a2) <= 1:
        return False

    if (a1.GetSymbol() == 'N' and a2.GetSymbol() == 'C') or (a1.GetSymbol() == 'C' and a2.GetSymbol() == 'N'):
        carbon_atom = a1 if a1.GetSymbol() == 'C' else a2
        for neighbor_bond in carbon_atom.GetBonds():
            if neighbor_bond.GetBondType() == Chem.BondType.DOUBLE:
                other_atom = neighbor_bond.GetOtherAtom(carbon_atom)
                if other_atom.GetSymbol() == 'O':
                    return False

    carbon_atom_list = []
    if a1.GetSymbol() == 'C':
        carbon_atom_list.append(a1)
    if a2.GetSymbol() == 'C':
        carbon_atom_list.append(a2)
    ori_bond_idx = bond.GetIdx()

    for neighbor_bond in a1.GetBonds():
        if neighbor_bond.GetBondType() == Chem.BondType.TRIPLE:
            return False
    for neighbor_bond in a2.GetBonds():
        if neighbor_bond.GetBondType() == Chem.BondType.TRIPLE:
            return False

    for carbon_atom in carbon_atom_list:
        bond_list = carbon_atom.GetBonds()
        is_rot = False
        if len(bond_list) != 4:
            is_rot = True
        other_atom_symbol = None
        for neighbor_bond in bond_list:
            if neighbor_bond.IsInRing():
                is_rot = True
            if neighbor_bond.GetIdx() == ori_bond_idx:
                continue
            other_atom = neighbor_bond.GetOtherAtom(carbon_atom)
            symbol = other_atom.GetSymbol()
            if other_atom_symbol is None:
                other_atom_symbol = symbol
                if other_atom_symbol not in ['C', 'F', 'Cl', 'Br']:
                    is_rot = True
            elif other_atom_symbol != symbol:
                is_rot = True
        if not is_rot:
            return False

    return True


def find_rotatable_bonds(mol):
    rb_data = []
    affected_set = set()
    for bond in mol.GetBonds():
        if is_rotatable_bond(bond):
            emol = Chem.EditableMol(mol)
            a1_idx, a2_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            emol.RemoveBond(a1_idx, a2_idx)
            modified_mol = emol.GetMol()
            frag_atom_indices = list(Chem.GetMolFrags(modified_mol))
            if len(frag_atom_indices) == 2:
                frag_sizes = [len(frag) for frag in frag_atom_indices]
                smaller_idx = 0 if frag_sizes[0] < frag_sizes[1] else 1
                rb_data.append([list(frag_atom_indices[smaller_idx]), a1_idx, a2_idx])
                affected_set.update(set(frag_atom_indices[smaller_idx]))
    rb_data = sorted(rb_data, key=lambda x: len(x[0]))
    atom_pairs = [[x[1], x[2]] for x in rb_data]
    affected_indices_list = [x[0] for x in rb_data]
    affected_indices = [np.ascontiguousarray(a, dtype=np.int32) for a in affected_indices_list]

    rot_affected_indices_list = []
    rot_affected_offsets = [0]
    for idx in affected_indices:
        rot_affected_indices_list.extend(idx)
        rot_affected_offsets.append(rot_affected_offsets[-1] + len(idx))
    atom_pairs_arr = np.ascontiguousarray(atom_pairs, dtype=np.int32)
    if atom_pairs_arr.size == 0:
        atom_pairs_arr = np.empty((0, 2), dtype=np.int32)
    return (atom_pairs_arr,
            np.ascontiguousarray(rot_affected_indices_list, dtype=np.int32),
            np.ascontiguousarray(rot_affected_offsets, dtype=np.int32))
