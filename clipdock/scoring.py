import csv

import numpy as np
from numba import njit
import xml.etree.ElementTree as ET

from rdkit import Chem

from clipdock.atom_type import (EscoreAtomType, GscoreAtomType,
                               EMBED_CENTERS, EMBED_SIGMAS, EMBED_PIS)
from clipdock.utils.calc import cal_gauss
from clipdock.utils.mol import get_intra_mask


@njit(error_model='numpy', fastmath=True, boundscheck=False, cache=True)
def _score_pair_exact(dx, dy, dz, type1, type2, e_min_parm, r_min_parm, r_d_parm):
    dist_sq = dx * dx + dy * dy + dz * dz
    if dist_sq <= 1e-20:
        return 0.0, 0.0, 0.0, 0.0

    n, m = 12.0, 6.0
    ratio_split = 1.3
    lj_factor1 = m / (n - m)
    lj_factor2 = n / (n - m)

    r = np.sqrt(dist_sq)
    r_min = r_min_parm[type1, type2]
    e_min = e_min_parm[type1, type2]
    r_d = r_d_parm[type1, type2]

    ratio = (r_min + r_d) / (r + r_d)
    d_smooth_d_ratio = 1.0
    if ratio > ratio_split:
        x = ratio - ratio_split
        ratio = ratio_split + x / (x * 3.0 + 1.0)
        denom = 3.0 * x + 1.0
        d_smooth_d_ratio = 1.0 / (denom * denom)

    ratio2 = ratio * ratio
    ratio4 = ratio2 * ratio2
    ratio6 = ratio4 * ratio2
    ratio12 = ratio6 * ratio6
    score = e_min * (lj_factor1 * ratio12 - lj_factor2 * ratio6)

    dE_d_ratio = e_min * (
        12.0 * lj_factor1 * ratio6 * ratio4 * ratio -
        6.0 * lj_factor2 * ratio4 * ratio
    )
    d_ratio_d_r = -(r_min + r_d) / ((r + r_d) * (r + r_d)) * d_smooth_d_ratio
    force_magnitude = dE_d_ratio * d_ratio_d_r / r
    return score, force_magnitude * dx, force_magnitude * dy, force_magnitude * dz


@njit(error_model='numpy', fastmath=True, boundscheck=False, cache=True)
def _calc_score_cell_list_with_grad_inplace(atom_types1, coords1, lig_ignore_idx,
                                            atom_types2, coords2, cell_origin,
                                            cell_dims, cell_starts, cell_counts,
                                            e_min_parm, r_min_parm, r_d_parm,
                                            max_cutoff, score_grad):
    score = 0.0
    max_cutoff_sq = max_cutoff * max_cutoff
    nx = cell_dims[0]
    ny = cell_dims[1]
    nz = cell_dims[2]

    for idx1 in range(coords1.shape[0]):
        score_grad[idx1, 0] = 0.0
        score_grad[idx1, 1] = 0.0
        score_grad[idx1, 2] = 0.0

    for idx1 in range(atom_types1.shape[0]):
        if idx1 == lig_ignore_idx:
            continue

        x1 = coords1[idx1, 0]
        y1 = coords1[idx1, 1]
        z1 = coords1[idx1, 2]
        cx = int(np.floor((x1 - cell_origin[0]) / max_cutoff))
        cy = int(np.floor((y1 - cell_origin[1]) / max_cutoff))
        cz = int(np.floor((z1 - cell_origin[2]) / max_cutoff))

        type1 = atom_types1[idx1]
        for ox in range(-1, 2):
            gx = cx + ox
            if gx < 0 or gx >= nx:
                continue
            for oy in range(-1, 2):
                gy = cy + oy
                if gy < 0 or gy >= ny:
                    continue
                for oz in range(-1, 2):
                    gz = cz + oz
                    if gz < 0 or gz >= nz:
                        continue
                    cell_id = (gx * ny + gy) * nz + gz
                    start = cell_starts[cell_id]
                    count = cell_counts[cell_id]
                    for offset in range(count):
                        idx2 = start + offset
                        dx = x1 - coords2[idx2, 0]
                        dx_sq = dx * dx
                        if dx_sq >= max_cutoff_sq:
                            continue
                        dy = y1 - coords2[idx2, 1]
                        partial_sq = dx_sq + dy * dy
                        if partial_sq >= max_cutoff_sq:
                            continue
                        dz = z1 - coords2[idx2, 2]
                        dist_sq = partial_sq + dz * dz
                        if dist_sq >= max_cutoff_sq:
                            continue

                        pair_score, fx, fy, fz = _score_pair_exact(
                            dx, dy, dz, type1, atom_types2[idx2],
                            e_min_parm, r_min_parm, r_d_parm
                        )
                        score += pair_score
                        score_grad[idx1, 0] += fx
                        score_grad[idx1, 1] += fy
                        score_grad[idx1, 2] += fz

    return score


@njit(error_model='numpy', fastmath=True, boundscheck=False, cache=True)
def _calc_score_cell_list(atom_types1, coords1, lig_ignore_idx,
                          atom_types2, coords2, cell_origin,
                          cell_dims, cell_starts, cell_counts,
                          e_min_parm, r_min_parm, r_d_parm,
                          max_cutoff):
    score = 0.0
    max_cutoff_sq = max_cutoff * max_cutoff
    nx = cell_dims[0]
    ny = cell_dims[1]
    nz = cell_dims[2]

    for idx1 in range(atom_types1.shape[0]):
        if idx1 == lig_ignore_idx:
            continue

        x1 = coords1[idx1, 0]
        y1 = coords1[idx1, 1]
        z1 = coords1[idx1, 2]
        cx = int(np.floor((x1 - cell_origin[0]) / max_cutoff))
        cy = int(np.floor((y1 - cell_origin[1]) / max_cutoff))
        cz = int(np.floor((z1 - cell_origin[2]) / max_cutoff))

        type1 = atom_types1[idx1]
        for ox in range(-1, 2):
            gx = cx + ox
            if gx < 0 or gx >= nx:
                continue
            for oy in range(-1, 2):
                gy = cy + oy
                if gy < 0 or gy >= ny:
                    continue
                for oz in range(-1, 2):
                    gz = cz + oz
                    if gz < 0 or gz >= nz:
                        continue
                    cell_id = (gx * ny + gy) * nz + gz
                    start = cell_starts[cell_id]
                    count = cell_counts[cell_id]
                    for offset in range(count):
                        idx2 = start + offset
                        dx = x1 - coords2[idx2, 0]
                        dx_sq = dx * dx
                        if dx_sq >= max_cutoff_sq:
                            continue
                        dy = y1 - coords2[idx2, 1]
                        partial_sq = dx_sq + dy * dy
                        if partial_sq >= max_cutoff_sq:
                            continue
                        dz = z1 - coords2[idx2, 2]
                        dist_sq = partial_sq + dz * dz
                        if dist_sq >= max_cutoff_sq:
                            continue

                        pair_score, _, _, _ = _score_pair_exact(
                            dx, dy, dz, type1, atom_types2[idx2],
                            e_min_parm, r_min_parm, r_d_parm
                        )
                        score += pair_score

    return score


@njit(error_model='numpy', fastmath=True, boundscheck=False, cache=True)
def _calc_score_intra_with_grad_inplace(atom_types, coords, lig_ignore_idx,
                                        e_min_parm, r_min_parm, r_d_parm,
                                        max_cutoff, lig_intra_mask, score_grad):
    score = 0.0
    max_cutoff_sq = max_cutoff * max_cutoff

    for idx in range(coords.shape[0]):
        score_grad[idx, 0] = 0.0
        score_grad[idx, 1] = 0.0
        score_grad[idx, 2] = 0.0

    for idx1 in range(atom_types.shape[0]):
        if idx1 == lig_ignore_idx:
            continue
        type1 = atom_types[idx1]
        x1 = coords[idx1, 0]
        y1 = coords[idx1, 1]
        z1 = coords[idx1, 2]
        for idx2 in range(idx1):
            if idx2 == lig_ignore_idx:
                continue
            if 1 <= lig_intra_mask[idx1, idx2] <= 4:
                continue

            dx = x1 - coords[idx2, 0]
            dx_sq = dx * dx
            if dx_sq >= max_cutoff_sq:
                continue
            dy = y1 - coords[idx2, 1]
            partial_sq = dx_sq + dy * dy
            if partial_sq >= max_cutoff_sq:
                continue
            dz = z1 - coords[idx2, 2]
            dist_sq = partial_sq + dz * dz
            if dist_sq >= max_cutoff_sq:
                continue

            pair_score, fx, fy, fz = _score_pair_exact(
                dx, dy, dz, type1, atom_types[idx2],
                e_min_parm, r_min_parm, r_d_parm
            )
            score += pair_score
            score_grad[idx1, 0] += fx
            score_grad[idx1, 1] += fy
            score_grad[idx1, 2] += fz
            score_grad[idx2, 0] -= fx
            score_grad[idx2, 1] -= fy
            score_grad[idx2, 2] -= fz

    return score


@njit(error_model='numpy', fastmath=True, boundscheck=False, cache=True)
def _calc_score_intra(atom_types, coords, lig_ignore_idx,
                      e_min_parm, r_min_parm, r_d_parm,
                      max_cutoff, lig_intra_mask):
    score = 0.0
    max_cutoff_sq = max_cutoff * max_cutoff

    for idx1 in range(atom_types.shape[0]):
        if idx1 == lig_ignore_idx:
            continue
        type1 = atom_types[idx1]
        x1 = coords[idx1, 0]
        y1 = coords[idx1, 1]
        z1 = coords[idx1, 2]
        for idx2 in range(idx1):
            if idx2 == lig_ignore_idx:
                continue
            if 1 <= lig_intra_mask[idx1, idx2] <= 4:
                continue

            dx = x1 - coords[idx2, 0]
            dx_sq = dx * dx
            if dx_sq >= max_cutoff_sq:
                continue
            dy = y1 - coords[idx2, 1]
            partial_sq = dx_sq + dy * dy
            if partial_sq >= max_cutoff_sq:
                continue
            dz = z1 - coords[idx2, 2]
            dist_sq = partial_sq + dz * dz
            if dist_sq >= max_cutoff_sq:
                continue

            pair_score, _, _, _ = _score_pair_exact(
                dx, dy, dz, type1, atom_types[idx2],
                e_min_parm, r_min_parm, r_d_parm
            )
            score += pair_score

    return score


class Scorer:
    def __init__(self, rec_mol, lig_mol, max_cutoff=8.0, pad_size=8.0,
                 keep_H=False, build_intra_mask=True):
        self.rec_mol = Chem.Mol(rec_mol)
        self.lig_mol = Chem.Mol(lig_mol)
        if not keep_H:
            self.rec_mol = Chem.RemoveAllHs(self.rec_mol)
            self.lig_mol = Chem.RemoveAllHs(self.lig_mol)
        self.rec_coords = np.ascontiguousarray(self.rec_mol.GetConformer().GetPositions(), dtype=np.float64)
        self.lig_coords = np.ascontiguousarray(self.lig_mol.GetConformer().GetPositions(), dtype=np.float64)
        self.max_cutoff = max_cutoff

        ref_mol = self.lig_mol
        ref_mol_coords = np.ascontiguousarray(ref_mol.GetConformer().GetPositions(), dtype=np.float64)
        ref_mol_coord_max = ref_mol_coords.max(axis=0)
        ref_mol_coord_min = ref_mol_coords.min(axis=0)
        self.pad_size = pad_size
        self.lig_box = np.ascontiguousarray(
            [[ref_mol_coord_min[0] - self.pad_size / 2, ref_mol_coord_max[0] + self.pad_size / 2],
             [ref_mol_coord_min[1] - self.pad_size / 2, ref_mol_coord_max[1] + self.pad_size / 2],
             [ref_mol_coord_min[2] - self.pad_size / 2, ref_mol_coord_max[2] + self.pad_size / 2]], dtype=np.float64)

        if build_intra_mask:
            self.lig_intra_mask, _ = get_intra_mask(self.lig_mol, include_dist=False)
        else:
            self.lig_intra_mask = None
        Chem.SanitizeMol(self.lig_mol)
        self.rot_plt = 0.0

    def is_outside_box(self, lig_coords):
        return ((lig_coords < self.lig_box[:, 0]) | (lig_coords > self.lig_box[:, 1])).any()


class EScorer(Scorer):
    def __init__(self, rec_mol, lig_mol, parmas_file, max_cutoff=8.0, pad_size=8.0):
        super(EScorer, self).__init__(rec_mol, lig_mol, max_cutoff, pad_size, keep_H=False)
        self.type_class = EscoreAtomType()
        self.rec_atom_types = np.ascontiguousarray(
            self.type_class.assign_atom_types(rec_mol, keep_H=False), dtype=np.int64
        )
        self.lig_atom_types = np.ascontiguousarray(
            self.type_class.assign_atom_types(lig_mol, keep_H=False), dtype=np.int64
        )

        self.is_h_donor_list = self.type_class.is_h_donor_list
        self.is_h_acceptor_list = self.type_class.is_h_acceptor_list

        (self.e_min, self.r_min, self.r_d, self.config_dict,
         self.uff_e_min, self.uff_r_min) = self.load_parmas(parmas_file)
        (self.rec_cell_origin, self.rec_cell_dims, self.rec_cell_starts,
         self.rec_cell_counts, self.rec_cell_atom_types,
         self.rec_cell_coords) = self.build_static_cell_list(
            self.rec_atom_types, self.rec_coords, self.max_cutoff
        )
        self._inter_grad_buffer = np.empty_like(self.lig_coords)
        self._intra_grad_buffer = np.empty_like(self.lig_coords)
        self._total_grad_buffer = np.empty_like(self.lig_coords)

    @staticmethod
    def build_static_cell_list(atom_types, coords, cutoff):
        atom_types = np.ascontiguousarray(atom_types, dtype=np.int64)
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        if coords.shape[0] == 0:
            origin = np.zeros(3, dtype=np.float64)
            dims = np.ones(3, dtype=np.int64)
            starts = np.zeros(1, dtype=np.int64)
            counts = np.zeros(1, dtype=np.int64)
            return origin, dims, starts, counts, atom_types, coords

        coord_min = coords.min(axis=0) - cutoff
        coord_max = coords.max(axis=0) + cutoff
        dims = np.floor((coord_max - coord_min) / cutoff).astype(np.int64) + 1
        dims = np.maximum(dims, 1)
        cell_ids = np.floor((coords - coord_min) / cutoff).astype(np.int64)
        flat_ids = (cell_ids[:, 0] * dims[1] + cell_ids[:, 1]) * dims[2] + cell_ids[:, 2]
        order = np.argsort(flat_ids, kind='stable')
        sorted_flat_ids = flat_ids[order]
        sorted_types = np.ascontiguousarray(atom_types[order], dtype=np.int64)
        sorted_coords = np.ascontiguousarray(coords[order], dtype=np.float64)
        cell_count = int(dims[0] * dims[1] * dims[2])
        starts = np.empty(cell_count, dtype=np.int64)
        counts = np.zeros(cell_count, dtype=np.int64)
        starts.fill(0)

        cursor = 0
        for cell_id in range(cell_count):
            starts[cell_id] = cursor
            while cursor < sorted_flat_ids.shape[0] and sorted_flat_ids[cursor] == cell_id:
                counts[cell_id] += 1
                cursor += 1
        return (np.ascontiguousarray(coord_min, dtype=np.float64),
                np.ascontiguousarray(dims, dtype=np.int64),
                np.ascontiguousarray(starts, dtype=np.int64),
                np.ascontiguousarray(counts, dtype=np.int64),
                sorted_types, sorted_coords)

    @staticmethod
    def calc_score_cell_list_with_grad_inplace(atom_types1, coords1, lig_ignore_idx,
                                               atom_types2, coords2, cell_origin,
                                               cell_dims, cell_starts, cell_counts,
                                               e_min_parm, r_min_parm, r_d_parm,
                                               max_cutoff, score_grad):
        ignore_idx = -1 if lig_ignore_idx is None else int(lig_ignore_idx)
        return _calc_score_cell_list_with_grad_inplace(
            np.ascontiguousarray(atom_types1, dtype=np.int64),
            np.ascontiguousarray(coords1, dtype=np.float64),
            ignore_idx,
            np.ascontiguousarray(atom_types2, dtype=np.int64),
            np.ascontiguousarray(coords2, dtype=np.float64),
            np.ascontiguousarray(cell_origin, dtype=np.float64),
            np.ascontiguousarray(cell_dims, dtype=np.int64),
            np.ascontiguousarray(cell_starts, dtype=np.int64),
            np.ascontiguousarray(cell_counts, dtype=np.int64),
            e_min_parm, r_min_parm, r_d_parm,
            float(max_cutoff),
            score_grad,
        )

    @staticmethod
    def calc_score_cell_list(atom_types1, coords1, lig_ignore_idx,
                             atom_types2, coords2, cell_origin,
                             cell_dims, cell_starts, cell_counts,
                             e_min_parm, r_min_parm, r_d_parm,
                             max_cutoff):
        ignore_idx = -1 if lig_ignore_idx is None else int(lig_ignore_idx)
        return _calc_score_cell_list(
            np.ascontiguousarray(atom_types1, dtype=np.int64),
            np.ascontiguousarray(coords1, dtype=np.float64),
            ignore_idx,
            np.ascontiguousarray(atom_types2, dtype=np.int64),
            np.ascontiguousarray(coords2, dtype=np.float64),
            np.ascontiguousarray(cell_origin, dtype=np.float64),
            np.ascontiguousarray(cell_dims, dtype=np.int64),
            np.ascontiguousarray(cell_starts, dtype=np.int64),
            np.ascontiguousarray(cell_counts, dtype=np.int64),
            e_min_parm, r_min_parm, r_d_parm,
            float(max_cutoff),
        )

    @staticmethod
    def calc_score_intra_with_grad_inplace(atom_types, coords, lig_ignore_idx,
                                           e_min_parm, r_min_parm, r_d_parm,
                                           max_cutoff, lig_intra_mask,
                                           score_grad):
        ignore_idx = -1 if lig_ignore_idx is None else int(lig_ignore_idx)
        return _calc_score_intra_with_grad_inplace(
            np.ascontiguousarray(atom_types, dtype=np.int64),
            np.ascontiguousarray(coords, dtype=np.float64),
            ignore_idx,
            e_min_parm, r_min_parm, r_d_parm,
            float(max_cutoff),
            np.ascontiguousarray(lig_intra_mask, dtype=np.int64),
            score_grad,
        )

    @staticmethod
    def calc_score_intra(atom_types, coords, lig_ignore_idx,
                         e_min_parm, r_min_parm, r_d_parm,
                         max_cutoff, lig_intra_mask):
        ignore_idx = -1 if lig_ignore_idx is None else int(lig_ignore_idx)
        return _calc_score_intra(
            np.ascontiguousarray(atom_types, dtype=np.int64),
            np.ascontiguousarray(coords, dtype=np.float64),
            ignore_idx,
            e_min_parm, r_min_parm, r_d_parm,
            float(max_cutoff),
            np.ascontiguousarray(lig_intra_mask, dtype=np.int64),
        )

    def load_parmas(self, parmas_file):
        tree = ET.parse(parmas_file)
        root = tree.getroot()
        radius_dict, deep_dict, config_dict = {}, {}, {}

        n = len(self.type_class.type_list)
        e_min = np.ones((n, n), dtype=np.float64)
        r_min = np.ones((n, n), dtype=np.float64)
        r_d = np.zeros((n, n), dtype=np.float64)
        uff_e_min = np.ones((n, n), dtype=np.float64)
        uff_r_min = np.ones((n, n), dtype=np.float64)

        ff_section = root.find('ForceField')
        for atom_type in ff_section.findall('Type'):
            name = atom_type.get('name')
            radius = float(atom_type.get('radius'))
            depth = float(atom_type.get('depth'))
            radius_dict[name] = radius
            deep_dict[name] = depth

        config_section = root.find('Config')
        for atom_type in config_section.findall('Type'):
            name = atom_type.get('name')
            value = float(atom_type.get('value'))
            config_dict[name] = value

        k = 1.0
        lj_r_min_list = np.ascontiguousarray(
            [radius_dict.get(k.split('_')[0]) for i, k in enumerate(self.type_class.type_list)],
            dtype=np.float64)
        lj_e_min_list = np.ascontiguousarray(
            [deep_dict.get(k.split('_')[0]) for i, k in enumerate(self.type_class.type_list)],
            dtype=np.float64)

        for type1_name in self.type_class.type_list:
            for type2_name in self.type_class.type_list:
                type1 = self.type_class.type_atom_to_id[type1_name]
                type2 = self.type_class.type_atom_to_id[type2_name]
                if type2 > type1:
                    break

                vdw_r = np.sqrt(lj_r_min_list[type1] * lj_r_min_list[type2])
                vdw_w = np.sqrt(lj_e_min_list[type1] * lj_e_min_list[type2])
                is_hbond = False
                if (self.is_h_donor_list[type1] and self.is_h_acceptor_list[type2]) or (
                        self.is_h_acceptor_list[type1] and self.is_h_donor_list[type2]):
                    is_hbond = True

                eps = 0.001
                parmas = [vdw_w, vdw_r, eps]
                if is_hbond:
                    hbond_w, hbond_r = vdw_w, vdw_r
                    has_N = type1_name[0] == 'N' or type2_name[0] == 'N'
                    has_O = type1_name[0] == 'O' or type2_name[0] == 'O'
                    if type1_name.split("_")[0] == 'Met' or type2_name.split("_")[0] == 'Met':
                        if has_N:
                            hbond_w, hbond_r = deep_dict["MB_Met_N"], radius_dict["MB_Met_N"]
                        elif has_O:
                            hbond_w, hbond_r = deep_dict["MB_Met_O"], radius_dict["MB_Met_O"]
                    elif type1_name.split("_")[0] in ['Cl', 'Br', 'I'] or type2_name.split("_")[0] in ['Cl', 'Br', 'I']:
                        if type1_name.split("_")[0] == 'Cl' or type2_name.split("_")[0] == 'Cl':
                            if has_O:
                                hbond_w, hbond_r = deep_dict["XB_Cl_O"], radius_dict["XB_Cl_O"]
                        elif type1_name.split("_")[0] == 'Br' or type2_name.split("_")[0] == 'Br':
                            if has_O:
                                hbond_w, hbond_r = deep_dict["XB_Br_O"], radius_dict["XB_Br_O"]
                        elif type1_name.split("_")[0] == 'I' or type2_name.split("_")[0] == 'I':
                            if has_O:
                                hbond_w, hbond_r = deep_dict["XB_I_O"], radius_dict["XB_I_O"]
                    else:
                        dict_key = None
                        if type1_name.split("_")[0] == 'N' and type2_name.split("_")[0] == 'N':
                            dict_key = "HB_N_N"
                        elif (type1_name.split("_")[0] == 'N' and type2_name.split("_")[0] == 'O') or (
                                type1_name.split("_")[0] == 'O' and type2_name.split("_")[0] == 'N'):
                            dict_key = "HB_N_O"
                        elif type1_name.split("_")[0] == 'O' and type2_name.split("_")[0] == 'O':
                            dict_key = "HB_O_O"
                        elif (type1_name.split("_")[0] == 'Np' and type2_name.split("_")[0] == 'N') or (
                                type1_name.split("_")[0] == 'N' and type2_name.split("_")[0] == 'Np'):
                            dict_key = "HB_Np_N"
                        elif (type1_name.split("_")[0] == 'Np' and type2_name.split("_")[0] == 'O') or (
                                type1_name.split("_")[0] == 'O' and type2_name.split("_")[0] == 'Np'):
                            dict_key = "HB_Np_O"
                        elif (type1_name.split("_")[0] == 'On' and type2_name.split("_")[0] == 'N') or (
                                type1_name.split("_")[0] == 'N' and type2_name.split("_")[0] == 'On'):
                            dict_key = "HB_On_N"
                        elif (type1_name.split("_")[0] == 'On' and type2_name.split("_")[0] == 'O') or (
                                type1_name.split("_")[0] == 'O' and type2_name.split("_")[0] == 'On'):
                            dict_key = "HB_On_O"
                        elif (type1_name.split("_")[0] == 'Np' and type2_name.split("_")[0] == 'On') or (
                                type1_name.split("_")[0] == 'On' and type2_name.split("_")[0] == 'Np'):
                            dict_key = "HB_Np_On"
                        if dict_key is not None:
                            hbond_w, hbond_r = deep_dict[dict_key], radius_dict[dict_key]
                    parmas = [hbond_w, hbond_r, eps]
                if type1_name == 'C' and type2_name == 'C':
                    k = 0.02 / parmas[0]
                e_min[type1][type2] = e_min[type2][type1] = parmas[0]
                r_min[type1][type2] = r_min[type2][type1] = parmas[1]
                r_d[type1][type2] = r_d[type2][type1] = parmas[2]
                uff_e_min[type1][type2] = uff_e_min[type2][type1] = vdw_w
                uff_r_min[type1][type2] = uff_r_min[type2][type1] = vdw_r
        return k * e_min, r_min, r_d, config_dict, k * uff_e_min, uff_r_min

    def calc_escore(self, lig_coords, lig_ignore_idx=None):
        inter = self.calc_score_cell_list(
            self.lig_atom_types, lig_coords, lig_ignore_idx,
            self.rec_cell_atom_types, self.rec_cell_coords,
            self.rec_cell_origin, self.rec_cell_dims,
            self.rec_cell_starts, self.rec_cell_counts,
            self.e_min, self.r_min, self.r_d,
            self.max_cutoff
        )
        intra = self.calc_score_intra(
            self.lig_atom_types, lig_coords, lig_ignore_idx,
            self.uff_e_min, self.uff_r_min, self.r_d,
            self.max_cutoff, self.lig_intra_mask
        )
        score = inter + intra
        return score

    def calc_escore_with_grad(self, lig_coords, lig_ignore_idx=None):
        inter = self.calc_score_cell_list_with_grad_inplace(
            self.lig_atom_types, lig_coords, lig_ignore_idx,
            self.rec_cell_atom_types, self.rec_cell_coords,
            self.rec_cell_origin, self.rec_cell_dims,
            self.rec_cell_starts, self.rec_cell_counts,
            self.e_min, self.r_min, self.r_d,
            self.max_cutoff, self._inter_grad_buffer
        )
        intra = self.calc_score_intra_with_grad_inplace(
            self.lig_atom_types, lig_coords, lig_ignore_idx,
            self.uff_e_min, self.uff_r_min, self.r_d,
            self.max_cutoff, self.lig_intra_mask, self._intra_grad_buffer
        )
        score = inter + intra
        self._total_grad_buffer[:, 0] = self._inter_grad_buffer[:, 0] + self._intra_grad_buffer[:, 0]
        self._total_grad_buffer[:, 1] = self._inter_grad_buffer[:, 1] + self._intra_grad_buffer[:, 1]
        self._total_grad_buffer[:, 2] = self._inter_grad_buffer[:, 2] + self._intra_grad_buffer[:, 2]
        return score, self._total_grad_buffer


class GScorer(Scorer):
    def __init__(self, rec_mol, lig_mol, parmas_file, max_cutoff=8.0, pad_size=8.0,
                 centers=EMBED_CENTERS, sigmas=EMBED_SIGMAS, pis=EMBED_PIS):
        super(GScorer, self).__init__(
            rec_mol, lig_mol, max_cutoff, pad_size, True, build_intra_mask=False
        )
        self.type_class = GscoreAtomType()
        self.rec_atom_types = np.ascontiguousarray(
            self.type_class.assign_atom_types(rec_mol, keep_H=True), dtype=np.int64
        )
        self.lig_atom_types = np.ascontiguousarray(
            self.type_class.assign_atom_types(lig_mol, keep_H=True), dtype=np.int64
        )

        self.dist_cutoff = 8
        self.gauss_num = len(centers)
        self.centers = np.ascontiguousarray(centers, dtype=np.float64)
        self.sigmas = np.ascontiguousarray(sigmas, dtype=np.float64)
        self.pis = np.ascontiguousarray(pis, dtype=np.float64)
        self.gauss_params = self.load_parmas(parmas_file)
        (self.rec_cell_origin, self.rec_cell_dims, self.rec_cell_starts,
         self.rec_cell_counts, self.rec_cell_atom_types,
         self.rec_cell_coords) = EScorer.build_static_cell_list(
            self.rec_atom_types, self.rec_coords, self.max_cutoff
        )

    def load_parmas(self, parmas_file):
        params = np.zeros((self.type_class.atom_type_num, self.type_class.atom_type_num, self.gauss_num),
                          dtype=np.float64)

        with open(parmas_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                lig_type = self.type_class.type_atom_to_id[row[0]]
                rec_type = self.type_class.type_atom_to_id[row[1]]
                params[lig_type, rec_type] = np.array([float(w) for w in row[2:]], dtype=np.float64)
        return params


    @staticmethod
    @njit(error_model='numpy', fastmath=True, boundscheck=False, cache=True)
    def _calc_score_cell_list(atom_types1, coords1, lig_ignore_idx,
                              atom_types2, coords2, cell_origin,
                              cell_dims, cell_starts, cell_counts,
                              centers, sigmas, pis, gauss_params,
                              max_cutoff):
        score = 0.0
        max_cutoff_sq = max_cutoff * max_cutoff
        nx = cell_dims[0]
        ny = cell_dims[1]
        nz = cell_dims[2]

        for idx1 in range(atom_types1.shape[0]):
            if idx1 == lig_ignore_idx:
                continue

            x1 = coords1[idx1, 0]
            y1 = coords1[idx1, 1]
            z1 = coords1[idx1, 2]
            cx = int(np.floor((x1 - cell_origin[0]) / max_cutoff))
            cy = int(np.floor((y1 - cell_origin[1]) / max_cutoff))
            cz = int(np.floor((z1 - cell_origin[2]) / max_cutoff))
            type1 = atom_types1[idx1]

            for ox in range(-1, 2):
                gx = cx + ox
                if gx < 0 or gx >= nx:
                    continue
                for oy in range(-1, 2):
                    gy = cy + oy
                    if gy < 0 or gy >= ny:
                        continue
                    for oz in range(-1, 2):
                        gz = cz + oz
                        if gz < 0 or gz >= nz:
                            continue
                        cell_id = (gx * ny + gy) * nz + gz
                        start = cell_starts[cell_id]
                        count = cell_counts[cell_id]
                        for offset in range(count):
                            idx2 = start + offset
                            dx = x1 - coords2[idx2, 0]
                            dx_sq = dx * dx
                            if dx_sq >= max_cutoff_sq:
                                continue
                            dy = y1 - coords2[idx2, 1]
                            partial_sq = dx_sq + dy * dy
                            if partial_sq >= max_cutoff_sq:
                                continue
                            dz = z1 - coords2[idx2, 2]
                            dist_sq = partial_sq + dz * dz
                            if dist_sq >= max_cutoff_sq:
                                continue

                            type2 = atom_types2[idx2]
                            d = np.sqrt(dist_sq)
                            pairwise_score = 0.0
                            for k in range(gauss_params.shape[-1]):
                                pairwise_score += (
                                    gauss_params[type1, type2, k] *
                                    pis[k] * cal_gauss(sigmas[k], centers[k], d)
                                )
                            score += pairwise_score
        return score

    @staticmethod
    def calc_score_cell_list(atom_types1, coords1, lig_ignore_idx,
                             atom_types2, coords2, cell_origin,
                             cell_dims, cell_starts, cell_counts,
                             centers, sigmas, pis, gauss_params,
                             max_cutoff):
        ignore_idx = -1 if lig_ignore_idx is None else int(lig_ignore_idx)
        return GScorer._calc_score_cell_list(
            np.ascontiguousarray(atom_types1, dtype=np.int64),
            np.ascontiguousarray(coords1, dtype=np.float64),
            ignore_idx,
            np.ascontiguousarray(atom_types2, dtype=np.int64),
            np.ascontiguousarray(coords2, dtype=np.float64),
            np.ascontiguousarray(cell_origin, dtype=np.float64),
            np.ascontiguousarray(cell_dims, dtype=np.int64),
            np.ascontiguousarray(cell_starts, dtype=np.int64),
            np.ascontiguousarray(cell_counts, dtype=np.int64),
            centers, sigmas, pis, gauss_params,
            float(max_cutoff),
        )

    def calc_gscore(self, lig_coords, lig_ignore_idx=None):
        return self.calc_score_cell_list(
            self.lig_atom_types, lig_coords, lig_ignore_idx,
            self.rec_cell_atom_types, self.rec_cell_coords,
            self.rec_cell_origin, self.rec_cell_dims,
            self.rec_cell_starts, self.rec_cell_counts,
            self.centers, self.sigmas, self.pis, self.gauss_params,
            self.max_cutoff
        )
