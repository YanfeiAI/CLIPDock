import os
import warnings

import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import PDBIO, PDBParser, Select, MMCIFParser
from rdkit import Chem

from clipdock.utils.mol import read_mol

warnings.filterwarnings("ignore", category=BiopythonWarning)


def remove_clashing_bonds(pdb_path, max_distance=2.0):
    extension = os.path.splitext(pdb_path)[1].lower()
    if extension in {'.cif', '.mmcif'}:
        mol = read_mol(pdb_path, sanitize=False, removeHs=False)
    else:
        mol = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to read receptor structure: {pdb_path}")
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
    extension = os.path.splitext(protein_path)[1].lower()
    if extension == '.pdb':
        parser = PDBParser(QUIET=True)
    elif extension in {'.cif', '.mmcif'}:
        parser = MMCIFParser(QUIET=True)
    else:
        return
    structure = parser.get_structure('protein', protein_path)

    selector = PocketSelector(coords, threshold, structure, remove_water, remove_cofactor, exclude_coords)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pocket_path, select=selector)


def read_coords(file_path):
    file_path = os.fspath(file_path)
    for sanitize in (True, False):
        mol = read_mol(file_path, try_mol=False, sanitize=sanitize, removeHs=False)
        if mol is None or mol.GetNumConformers() == 0:
            continue
        heavy = np.array([atom.GetAtomicNum() != 1 for atom in mol.GetAtoms()], dtype=bool)
        coords = mol.GetConformer().GetPositions()[heavy]
        if coords.size > 0 and np.isfinite(coords).all():
            return coords
    raise ValueError(f"Failed to read reference coordinates: {file_path}")


def extract_pocket(ref_ligand_path, protein_path, out_pocket_path, remove_water=False, remove_cofactor=False, threshold=8.0, exclude_coords=None):
    coords = read_coords(ref_ligand_path)
    extract_pocket_from_coords(coords, protein_path, out_pocket_path, remove_water, remove_cofactor, threshold, exclude_coords)
