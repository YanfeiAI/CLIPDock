import re
import warnings

import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import PDBIO, PDBParser, Select, MMCIFParser
from rdkit import Chem

from clipdock.utils.mol import read_mol

warnings.filterwarnings("ignore", category=BiopythonWarning)


def remove_clashing_bonds(pdb_path, max_distance=2.0):
    mol = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False)
    edit_mol = Chem.RWMol(mol)
    conf = mol.GetConformer()

    bonds_to_remove = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)
        info_i = atom_i.GetPDBResidueInfo()
        info_j = atom_j.GetPDBResidueInfo()

        if info_i and info_j and info_i.GetResidueNumber() != info_j.GetResidueNumber():
            distance = Chem.rdMolTransforms.GetBondLength(conf, i, j)
            if distance < max_distance:
                if not (atom_i.GetSymbol() == 'S' and atom_j.GetSymbol() == 'S'):
                    if not is_amide_bond(atom_i, atom_j, info_i, info_j):
                        bonds_to_remove.append((i, j))

    for i, j in bonds_to_remove:
        edit_mol.RemoveBond(i, j)

    return edit_mol.GetMol()


def is_amide_bond(atom1, atom2, info1, info2):
    if (atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'N') or \
            (atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'C'):
        carbon_name = info1.GetName().strip() if atom1.GetSymbol() == 'C' else info2.GetName().strip()
        nitrogen_name = info1.GetName().strip() if atom1.GetSymbol() == 'N' else info2.GetName().strip()
        if carbon_name == "C" and nitrogen_name == "N":
            return True
        residue1 = info1.GetResidueNumber()
        residue2 = info2.GetResidueNumber()
        if abs(residue1 - residue2) == 1:
            return True
    return False


class PocketSelector(Select):
    def __init__(self, ref_coords, threshold, structure, remove_water=False, remove_cofactor=False, exclude_coords=None):
        super().__init__()
        self.ref_coords = ref_coords
        self.threshold = threshold
        self.keep_residues = set()
        self.remove_water = remove_water
        self.remove_cofactor = remove_cofactor
        self.exclude_coords = exclude_coords

        for model in structure:
            for chain in model:
                for residue in chain:
                    if self.remove_water and residue.resname.strip() in ["HOH", "WAT"]:
                        continue

                    if self.remove_cofactor and residue.id[0].strip() != '' and residue.resname.strip() not in ["HOH", "WAT"]:
                        continue

                    for atom in residue:
                        if self.exclude_coords is not None:
                            delta = self.exclude_coords - atom.coord.reshape(1, 3)
                            dists = np.linalg.norm(delta, axis=1)
                            if np.any(dists <= 0.01):
                                break
                        delta = self.ref_coords - atom.coord.reshape(1, 3)
                        dists = np.linalg.norm(delta, axis=1)
                        if np.any(dists <= self.threshold):
                            self.keep_residues.add((chain.id, residue.id))
                            break

    def accept_residue(self, residue):
        return (residue.parent.id, residue.id) in self.keep_residues


def extract_pocket_from_coords(coords, protein_path, out_pocket_path, remove_water=False, remove_cofactor=False, threshold=8.0, exclude_coords=None):
    if protein_path.endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    elif protein_path.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        return
    structure = parser.get_structure('protein', protein_path)

    selector = PocketSelector(coords, threshold, structure, remove_water, remove_cofactor, exclude_coords)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pocket_path, select=selector)


def extract_coords_from_mol2(mol2_path):
    coords = []
    atom_pattern = re.compile(r'^\s*\d+\s+\S+\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)')

    try:
        with open(mol2_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.readlines()
        in_atom_section = False
        for line in content:
            line = line.strip()
            if line.startswith('@<TRIPOS>ATOM'):
                in_atom_section = True
                continue
            elif line.startswith('@<TRIPOS>'):
                in_atom_section = False
                continue
            if in_atom_section:
                match = atom_pattern.match(line)
                if match:
                    try:
                        x = float(match.group(1))
                        y = float(match.group(2))
                        z = float(match.group(3))
                        coords.append([x, y, z])
                    except ValueError:
                        continue
                else:
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            x = float(parts[2])
                            y = float(parts[3])
                            z = float(parts[4])
                            coords.append([x, y, z])
                        except (ValueError, IndexError):
                            continue
        return np.array(coords)
    except Exception:
        pass
    return np.array([])


def extract_coords_from_sdf(sdf_path):
    coords = []
    in_molecule = False
    atom_count = 0
    read_atoms = False
    try:
        with open(sdf_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('$$$$') and in_molecule:
                    break
                if read_atoms and atom_count > 0:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            z = float(parts[2])
                            coords.append([x, y, z])
                            atom_count -= 1
                        except ValueError:
                            continue
                    if atom_count == 0:
                        read_atoms = False
                elif in_molecule and len(line.split()) == 11:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            atom_count = int(parts[0])
                            if atom_count > 0:
                                read_atoms = True
                                next(f)
                        except ValueError:
                            continue
                elif not in_molecule and len(line) > 0 and not line.startswith(' '):
                    in_molecule = True
        return np.array(coords)
    except Exception:
        pass
    return np.array([])


def read_coords(file_path):
    ref_mol = read_mol(file_path)
    coords = None
    if ref_mol is None:
        if file_path.endswith('.mol2'):
            coords = extract_coords_from_mol2(file_path)
        elif file_path.endswith('.sdf'):
            coords = extract_coords_from_sdf(file_path)
    else:
        coords = np.array([ref_mol.GetConformer().GetAtomPosition(i) for i in range(ref_mol.GetNumAtoms())])
    return coords


def extract_pocket(ref_ligand_path, protein_path, out_pocket_path, remove_water=False, remove_cofactor=False, threshold=8.0, exclude_coords=None):
    coords = read_coords(ref_ligand_path)
    if coords is None:
        raise ValueError('coords is None')
    extract_pocket_from_coords(coords, protein_path, out_pocket_path, remove_water, remove_cofactor, threshold, exclude_coords)
