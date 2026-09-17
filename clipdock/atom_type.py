import numpy as np
from rdkit import Chem

from clipdock.utils.constant import METALS
from clipdock.utils.mol import cal_atom_charge


def is_imine_nitrogen(nitrogen_atom):
    if (nitrogen_atom.GetSymbol() != 'N' or
            nitrogen_atom.GetFormalCharge() != 0 or
            str(nitrogen_atom.GetHybridization()) != 'SP2'):
        return False

    if sum(n.GetAtomicNum() != 1 for n in nitrogen_atom.GetNeighbors()) != 0:
        return False

    for bond in nitrogen_atom.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            neighbor = bond.GetOtherAtom(nitrogen_atom)
            if neighbor.GetSymbol() == 'C':
                return True
    return False


def is_imidazole_nitrogen(nitrogen_atom):
    if (nitrogen_atom.GetSymbol() != 'N' or
            nitrogen_atom.GetFormalCharge() != 0 or
            str(nitrogen_atom.GetHybridization()) != 'SP2'):
        return False

    mol = nitrogen_atom.GetOwningMol()
    pattern = Chem.MolFromSmarts("[c]1[n][c][n][c]1")
    matches = mol.GetSubstructMatches(pattern)
    for match in matches:
        if nitrogen_atom.GetIdx() == match[1]:
            return True
        if nitrogen_atom.GetIdx() == match[3]:
            return True
    return False


class AtomType:
    def __init__(self):
        self.type_list = None
        self.type_atom_to_id = None

    def assign_atom_type(self, atom):
        pass

    def assign_atom_type_id(self, atom):
        atom_type = self.assign_atom_type(atom)
        return self.type_atom_to_id.get(atom_type, self.type_atom_to_id['U'])

    def assign_atom_types(self, mol, keep_H=True):
        if keep_H:
            atom_types = [self.type_atom_to_id[self.assign_atom_type(atom)] for atom in mol.GetAtoms()]
        else:
            atom_types = [self.type_atom_to_id[self.assign_atom_type(atom)] for atom in mol.GetAtoms() if
                          atom.GetSymbol() != 'H']
        return np.ascontiguousarray(atom_types, dtype=np.int32)


class EscoreAtomType(AtomType):
    def __init__(self):
        super().__init__()
        self.type_list = ['C', 'F', 'Cl', 'Br', 'I', 'N_D', 'Np_D', 'O_D',  # [2, 7]
                          'Met_DA', 'N_DA', 'O_DA',  # [8, 10]
                          'N_A', 'O_A', 'On_A',  # [11, 13]
                          'N', 'Np', 'O', 'On', 'S', 'P', 'U', 'H', 'C_cov', 'S_H'  # [14, 21]
                          ]

        self.atom_type_num = len(self.type_list)
        self.type_id_to_atom = {i: k for i, k in enumerate(self.type_list)}
        self.type_atom_to_id = {k: i for i, k in enumerate(self.type_list)}
        self.is_h_donor_list = np.ascontiguousarray([False for i in range(len(self.type_list))], dtype=bool)
        self.is_h_donor_list[2: 10 + 1] = True
        self.is_h_acceptor_list = np.ascontiguousarray([False for i in range(len(self.type_list))], dtype=bool)
        self.is_h_acceptor_list[8: 13 + 1] = True

    def assign_atom_type(self, atom):
        symbol = atom.GetSymbol()
        is_aromatic = atom.GetIsAromatic()
        h_num = sum(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())
        if symbol == 'H':
            return 'H'
        elif symbol == 'C':
            return 'C'
        elif symbol == 'N':
            exp_charge, charge, charge_part = cal_atom_charge(atom)
            ring5 = atom.IsInRingSize(5)
            ring6 = atom.IsInRingSize(6)
            donor = h_num > 0

            if charge > 0:
                if donor:
                    return 'Np_D'
                return 'Np'

            if is_imine_nitrogen(atom):
                if donor:
                    return 'N_DA'
                else:
                    return 'N_A'

            if is_imidazole_nitrogen(atom):
                return 'N_DA'

            if is_aromatic and ((ring5 and not ring6) or (ring6 and not ring5)):
                if donor:
                    return 'N_D'
                return 'N_A'

            if donor:
                return 'N_D'
            return 'N'
        elif symbol == 'O':
            exp_charge, charge, charge_part = cal_atom_charge(atom)
            if charge == 0 and (h_num == 2 or len(atom.GetNeighbors()) == 0):
                return 'O_DA'
            donor = h_num > 0

            lone_pairs = (6 - charge - atom.GetTotalValence()) // 2
            acceptor = lone_pairs > 0

            if charge < 0:
                if acceptor:
                    return 'On_A'
                else:
                    return 'On'
            if donor and acceptor:
                return 'O_DA'
            elif donor:
                return 'O_D'
            elif acceptor:
                return 'O_A'
            return 'O'
        elif symbol == 'S':
            if h_num > 0:
                return 'S_H'
            return 'S'
        elif symbol in ['P', 'F', 'Cl', 'Br', 'I']:
            return symbol
        elif symbol in METALS:
            return 'Met_DA'
        else:
            return 'U'


def calc_o_group(atom):
    symbol = atom.GetSymbol()
    if symbol == 'O' and atom.GetFormalCharge() == 0 and str(atom.GetHybridization()) == 'SP2':
        for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                neighbor = bond.GetOtherAtom(atom)
                if neighbor.GetSymbol() == 'C':
                    carbon_atom = neighbor
                    has_nitrogen = False
                    has_hydrogen = False

                    for carbon_bond in carbon_atom.GetBonds():
                        if carbon_bond.GetBondType() == Chem.BondType.SINGLE:
                            other_neighbor = carbon_bond.GetOtherAtom(carbon_atom)
                            if other_neighbor.GetSymbol() == 'N':
                                has_nitrogen = True
                            elif other_neighbor.GetSymbol() == 'H':
                                has_hydrogen = True

                    if has_nitrogen:
                        return "O_AMID"
                    elif has_hydrogen:
                        return "O_ALD"
                    else:
                        return "O_CAR"
    return None


def is_imine_n(atom):
    if (atom.GetSymbol() != 'N' or
            atom.GetFormalCharge() != 0 or
            str(atom.GetHybridization()) != 'SP2'):
        return False

    for bond in atom.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            neighbor = bond.GetOtherAtom(atom)
            if neighbor.GetSymbol() == 'C':
                return True
    return False

class GscoreAtomType(AtomType):
    def __init__(self):
        super().__init__()
        self.type_list = ['H', 'H_N', 'H_O', 'H_S', 'H_ba',
                          'C', 'C_a', 'C_SP2', 'C_SP',
                          'F', 'F_ba',
                          'Cl', 'Cl_ba', 'Br', 'Br_ba', 'I', 'I_ba',
                          'N_D', 'N_a_D', 'N_ba_D', 'Np_D', 'N_SP2_D', 'N_AMID_D', 'Np_RES_D',
                          'N_DA', 'N_a_DA', 'O_DA', 'O_ba_DA', 'Met_DA',
                          'N_A', 'N_a_A',
                          'O_A', 'O_a_A', 'On_A', 'O_SP2_A', 'O_AMID', 'O_CAR', 'O_ALD',
                          'N', 'Np', 'N_SP',
                          'O', 'On',
                          'S', 'S_a', 'S_H',
                          'P', 'U',
                          ]

        self.atom_type_num = len(self.type_list)
        self.type_id_to_atom = {i: k for i, k in enumerate(self.type_list)}
        self.type_atom_to_id = {k: i for i, k in enumerate(self.type_list)}

    def assign_atom_type(self, atom):
        symbol = atom.GetSymbol()
        is_aromatic = atom.GetIsAromatic()
        h_num = sum(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())
        hybrid = str(atom.GetHybridization())

        if symbol == 'H':
            bond = atom.GetBonds()[0]
            neighbor = bond.GetOtherAtom(atom)
            neighbor_symbol = neighbor.GetSymbol()
            if neighbor_symbol != 'H':
                if neighbor_symbol == 'N':
                    return 'H_N'
                elif neighbor_symbol == 'O':
                    return 'H_O'
                elif neighbor_symbol == 'S':
                    return 'H_S'
                if neighbor.GetIsAromatic():
                    return 'H_ba'
            return 'H'
        if symbol == 'C':
            if is_aromatic:
                return 'C_a'
            if hybrid in ['SP2', 'SP']:
                return f'C_{hybrid}'
            return 'C'
        elif symbol == 'N':
            exp_charge, charge, charge_part = cal_atom_charge(atom)
            ring5 = atom.IsInRingSize(5)
            ring6 = atom.IsInRingSize(6)
            donor = h_num > 0

            if charge > 0:
                if donor:
                    if charge_part == 0 or charge_part == 1:
                        return 'Np_RES_D'
                    return 'Np_D'
                return 'Np'

            if is_imine_n(atom):
                if donor:
                    return 'N_DA'
                else:
                    return 'N_A'

            if is_imidazole_nitrogen(atom):
                return 'N_a_DA'

            if is_aromatic and ((ring5 and not ring6) or (ring6 and not ring5)):
                if donor:
                    return 'N_a_D'
                return 'N_a_A'

            if donor:
                if is_aromatic:
                    return 'N_a_D'
                bonds = [bond for bond in atom.GetBonds() if bond.GetOtherAtom(atom).GetAtomicNum() != 1]
                for i in range(len(bonds)):
                    bond = atom.GetBonds()[i]
                    neighbor = bond.GetOtherAtom(atom)
                    if neighbor.GetIsAromatic():
                        return 'N_ba_D'
                for bond in atom.GetBonds():
                    if bond.GetBondType() == Chem.BondType.SINGLE:
                        neighbor = bond.GetOtherAtom(atom)
                        if neighbor.GetSymbol() == 'C':
                            carbon_atom = neighbor
                            for carbon_bond in carbon_atom.GetBonds():
                                if (carbon_bond.GetBondType() == Chem.BondType.DOUBLE and
                                        carbon_bond.GetOtherAtom(carbon_atom).GetSymbol() == 'O'):
                                    return "N_AMID_D"
                if hybrid == 'SP2':
                    return 'N_SP2_D'
                return 'N_D'
            if hybrid == 'SP':
                return 'N_SP'
            return 'N'
        elif symbol == 'O':
            exp_charge, charge, charge_part = cal_atom_charge(atom)
            if charge == 0 and (h_num == 2 or len(atom.GetNeighbors()) == 0):
                return 'O_DA'
            donor = h_num > 0

            lone_pairs = (6 - charge - atom.GetTotalValence()) // 2
            acceptor = lone_pairs > 0

            if charge < 0:
                if acceptor:
                    return 'On_A'
                else:
                    return 'On'
            if donor and acceptor:
                bonds = [bond for bond in atom.GetBonds() if bond.GetOtherAtom(atom).GetAtomicNum() != 1]
                if len(bonds) == 1:
                    bond = atom.GetBonds()[0]
                    neighbor = bond.GetOtherAtom(atom)
                    if neighbor.GetIsAromatic():
                        return 'O_ba_DA'
                return 'O_DA'
            elif donor:
                return 'O_D'
            elif acceptor:
                if is_aromatic:
                    return 'O_a_A'
                type_name = calc_o_group(atom)
                if type_name is not None:
                    return type_name
                if hybrid == 'SP2':
                    return 'O_SP2_A'
                return 'O_A'
            return 'O'
        elif symbol == 'S':
            if is_aromatic:
                return 'S_a'
            if h_num > 0:
                return 'S_H'
            return 'S'
        elif symbol == 'P':
            return 'P'
        elif symbol in ['F', 'Cl', 'Br', 'I']:
            bonds = [bond for bond in atom.GetBonds() if bond.GetOtherAtom(atom).GetAtomicNum() != 1]
            if len(bonds) == 1:
                bond = atom.GetBonds()[0]
                neighbor = bond.GetOtherAtom(atom)
                if neighbor.GetIsAromatic():
                    return f'{symbol}_ba'
            return symbol
        elif symbol in METALS:
            return 'Met_DA'
        else:
            return 'U'


dist_cutoff = 8
EMBED_GAUSS_NUM = 8
reso = dist_cutoff / EMBED_GAUSS_NUM
EMBED_CENTERS = np.ascontiguousarray([reso * (i + 0.5) for i in range(EMBED_GAUSS_NUM)], dtype=np.float32)
EMBED_SIGMAS = np.ascontiguousarray([reso for i in range(len(EMBED_CENTERS))], dtype=np.float32)
EMBED_PIS = np.ascontiguousarray([10.0 / (c ** 2) for c in EMBED_CENTERS], dtype=np.float32)

type_class = GscoreAtomType()
EMBED_ATOM_TYPE_NUM = type_class.atom_type_num
