import random
import numpy as np
from numpy.random import Generator, PCG64

from rdkit import Chem
from scipy.optimize import minimize

from clipdock.scoring import EScorer, GScorer
from clipdock.utils.rot import find_rotatable_bonds
from clipdock.utils.trans import (apply_transformations, apply_transformations_inplace,
                                  project_transform_gradients_inplace)
from clipdock.utils.constant import GAUSS_PARAMS_PATH, DOCK_SCORE_K, DOCKING_PARAMS_PATH, DIST_CUTOFF, \
    PAD_SIZE
from clipdock.utils.mol import mol_gen_3d, protonate_mol


class PoseOptimizer:
    def __init__(self, rec_mol_h, lig_mol_h, params_file, lig_ff_num=0, seed=0, max_iter=64,
                 dist_cutoff=8.0, pad_size=8.0, gscore_params_path=GAUSS_PARAMS_PATH):
        gscore_params_path = gscore_params_path or GAUSS_PARAMS_PATH
        self.obj_score_base = 1000
        self.max_ftol = 3e-5
        self.min_ftol = 3e-5
        self.init_rate = 0.6
        self.offspring_rate = 0.15
        self.tor_rate = 0.3
        self.trans_sigma = 0.3

        self.lig_mol_h = Chem.Mol(lig_mol_h)
        self.lig_mol = Chem.RemoveAllHs(Chem.Mol(lig_mol_h))
        self.rec_mol_woh = Chem.RemoveAllHs(rec_mol_h)
        self.scorer = EScorer(rec_mol_h, self.lig_mol_h, params_file, dist_cutoff, pad_size)
        self.gauss_scorer = GScorer(rec_mol_h, protonate_mol(Chem.Mol(self.lig_mol)), gscore_params_path, dist_cutoff,
                                    pad_size)
        (self.rot_atom_pairs, self.rot_affected_indices,
         self.rot_affected_offsets) = find_rotatable_bonds(self.lig_mol)
        self.rot_num = len(self.rot_atom_pairs)
        self.seed = seed
        self.rng = Generator(PCG64(seed))
        self.max_iter = max_iter
        self.step_sizes = np.ascontiguousarray([0.5, np.pi / 6.0, np.pi], dtype=np.float64)

        ref_mol = Chem.RemoveHs(Chem.Mol(lig_mol_h))
        ref_mol_coords = np.ascontiguousarray(ref_mol.GetConformer().GetPositions(), dtype=np.float64)
        ref_mol_coord_max = ref_mol_coords.max(axis=0)
        ref_mol_coord_min = ref_mol_coords.min(axis=0)
        self.lig_size = np.ascontiguousarray([[ref_mol_coord_min[0], ref_mol_coord_max[0]],
                                              [ref_mol_coord_min[1], ref_mol_coord_max[1]],
                                              [ref_mol_coord_min[2], ref_mol_coord_max[2]]], dtype=np.float64)
        self.lig_ff_num = lig_ff_num
        self.ori_coords_arr, self.anchor_arr = self.gen_lig_ff_coords(self.lig_mol_h, self.lig_ff_num)
        self.ori_coords_len = len(self.ori_coords_arr)
        self.ori_coords = self.ori_coords_arr[0]
        self.anchor = self.anchor_arr[0]
        atom_count = self.ori_coords.shape[0]
        self.coords_buffer = np.empty((atom_count, 3), dtype=np.float64)
        self.pose_internal_coords_buffer = np.empty((atom_count, 3), dtype=np.float64)
        self.pose_internal_grad_buffer = np.empty((atom_count, 3), dtype=np.float64)
        self.pose_rotation_matrix_buffer = np.empty((3, 3), dtype=np.float64)
        self.grad_buffer = np.empty(6 + self.rot_num, dtype=np.float64)

    def refresh_lig(self):
        self.ori_coords_arr, self.anchor_arr = self.gen_lig_ff_coords(self.lig_mol_h, self.lig_ff_num)
        self.seed = random.SystemRandom().randrange(0, 2147483647)
        self.rng = Generator(PCG64(self.seed))

    @staticmethod
    def gen_lig_ff_coords(lig_mol_h, lig_ff_num):
        ori_coords_list = []

        for i in range(lig_ff_num):
            try:
                new_lig = mol_gen_3d(lig_mol_h)
                new_lig = Chem.RemoveAllHs(new_lig)
                ori_coords_list.append(new_lig.GetConformer().GetPositions())
                break
            except Exception:
                pass

        if len(ori_coords_list) == 0:
            lig_mol = Chem.Mol(lig_mol_h)
            lig_mol = Chem.RemoveAllHs(lig_mol)
            ori_coords_list.append(lig_mol.GetConformer().GetPositions())

        ori_coords_arr = np.ascontiguousarray(ori_coords_list, dtype=np.float64)
        anchor_arr = np.ascontiguousarray(
            [np.mean(coords, axis=0) for coords in ori_coords_list], dtype=np.float64)
        return ori_coords_arr, anchor_arr

    def init_rigid_transform(self, x, trans_sigma=0.3):
        x[3:6] = self.rng.uniform(-np.pi, np.pi, 3)

        min_bound = [min((self.scorer.lig_box[i][0] - self.lig_size[i][0]), 0) for i in range(3)]
        max_bound = [max((self.scorer.lig_box[i][1] - self.lig_size[i][1]), 0) for i in range(3)]

        center = [(min_bound[i] + max_bound[i]) / 2 for i in range(3)]
        box_size = [max_bound[i] - min_bound[i] for i in range(3)]
        x[0:3] = [
            np.clip(self.rng.normal(center[i], trans_sigma * box_size[i]),
                    min_bound[i], max_bound[i])
            for i in range(3)
        ]
        return x

    def random_init_x(self, tor_rate=0.3, trans_sigma=0.3):
        x = np.ascontiguousarray(np.zeros(6 + self.rot_num), dtype=np.float64)
        x = self.init_rigid_transform(x, trans_sigma=trans_sigma)
        x = self.mutate_tor(x, tor_rate)
        return x

    def mutate_tor(self, x, tor_rate=0.3):
        n_tor = int(round(self.rng.normal(loc=(len(x) - 6) * tor_rate, scale=0.5)))
        n_tor = min(max(n_tor, 0), len(x) - 6)
        torsion_n = 6
        if len(x) > 6 and n_tor > 0 and tor_rate > 0:
            idxs = self.rng.choice(range(6, len(x)), size=n_tor, replace=False)
            for idx in idxs:
                rot_list = [2 * np.pi * n / torsion_n for n in range(torsion_n)]
                rot_list = [a - 2 * np.pi if a > np.pi else a for a in rot_list]
                x[idx] = rot_list[self.rng.integers(0, len(rot_list))]
        return x

    def objective_with_grad(self, x):
        apply_transformations_inplace(
            x, self.ori_coords, self.anchor,
            self.rot_atom_pairs,
            self.rot_affected_indices,
            self.rot_affected_offsets,
            self.coords_buffer,
            self.pose_rotation_matrix_buffer
        )
        score, atom_grad = self.scorer.calc_escore_with_grad(self.coords_buffer)
        score += self.obj_score_base
        project_transform_gradients_inplace(
            x, self.ori_coords, self.anchor,
            self.rot_atom_pairs,
            self.rot_affected_indices,
            self.rot_affected_offsets,
            atom_grad,
            self.grad_buffer,
            self.pose_internal_coords_buffer,
            self.pose_internal_grad_buffer,
            self.pose_rotation_matrix_buffer
        )
        return score, self.grad_buffer

    def BFGS(self, init_x=None, ftol=None):
        if ftol is None:
            ftol = self.min_ftol
        if init_x is None:
            init_x = np.ascontiguousarray(np.zeros(6 + self.rot_num), dtype=np.float64)
        return minimize(self.objective_with_grad, init_x, jac=True, method='L-BFGS-B',
                        options={'ftol': ftol})

    def minimize_pose(self):
        result = self.BFGS()
        coords = apply_transformations(
            result.x, self.ori_coords, self.anchor,
            self.rot_atom_pairs, self.rot_affected_indices,
            self.rot_affected_offsets,
        )
        mol = Chem.Mol(self.lig_mol)
        conf = mol.GetConformer()
        for atom_idx in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(atom_idx, coords[atom_idx])
        mol = protonate_mol(mol)
        gscore = self.gauss_scorer.calc_gscore(mol.GetConformer().GetPositions())
        score = (self.scorer.calc_escore(coords) + gscore + self.scorer.rot_plt) * DOCK_SCORE_K
        return [[score, coords]]

    def gauss_inter(self, bfgs_result):
        x = bfgs_result.x
        coords = apply_transformations(x, self.ori_coords, self.anchor,
                                       self.rot_atom_pairs,
                                       self.rot_affected_indices,
                                       self.rot_affected_offsets
                                       )
        mol = Chem.Mol(self.lig_mol)
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, coords[i])
        mol = protonate_mol(mol)
        gauss = self.gauss_scorer.calc_gscore(mol.GetConformer().GetPositions())
        score = gauss + bfgs_result.fun - self.obj_score_base
        return score

    def evolve(self, pose_num=1, use_gauss=True):
        max_ftol, min_ftol = self.max_ftol, self.min_ftol

        init_rate = self.init_rate
        offspring_rate = self.offspring_rate
        tor_rate = self.tor_rate
        trans_sigma = self.trans_sigma

        population = []
        it = self.max_iter
        init_num = max(int(np.round(it * init_rate)), 1)
        ori_idx = 0
        for i in range(init_num):
            self.ori_coords, self.anchor = self.ori_coords_arr[ori_idx], self.anchor_arr[ori_idx]
            new_x = self.random_init_x(tor_rate=tor_rate, trans_sigma=trans_sigma)

            bfgs_result = self.BFGS(new_x, ftol=max_ftol)
            new_x = bfgs_result.x
            new_score = bfgs_result.fun - self.obj_score_base

            if use_gauss:
                new_score = self.gauss_inter(bfgs_result)

            population.append([new_score, new_x, ori_idx])
            ori_idx = (ori_idx + 1) % self.ori_coords_len
        it -= init_num

        history = population.copy()
        history.sort(key=lambda x: x[0])
        while it and len(population) > 0:
            select_num = min(max(int(it * offspring_rate), 1), it)
            select_num = min(len(population), select_num)
            population.sort(key=lambda x: x[0])
            population = population[: select_num]
            for i in range(select_num):
                ori_idx = population[i][2]
                self.ori_coords, self.anchor = self.ori_coords_arr[ori_idx], self.anchor_arr[ori_idx]
                new_x = population[i][1].copy()
                new_x = self.mutate_tor(new_x, tor_rate=tor_rate)

                bfgs_result = self.BFGS(new_x, ftol=min_ftol)
                new_x = bfgs_result.x
                new_score = bfgs_result.fun - self.obj_score_base

                if use_gauss:
                    new_score = self.gauss_inter(bfgs_result)

                population.append([new_score, new_x, ori_idx])
                history.append([new_score, new_x, ori_idx])
            it -= select_num

        history.sort(key=lambda x: x[0])
        history = history[:pose_num]
        pose_list = []
        for i in range(len(history)):
            ori_idx = history[i][2]
            self.ori_coords, self.anchor = self.ori_coords_arr[ori_idx], self.anchor_arr[ori_idx]
            coords = apply_transformations(
                history[i][1], self.ori_coords, self.anchor,
                self.rot_atom_pairs,
                self.rot_affected_indices,
                self.rot_affected_offsets
            )
            score = history[i][0]
            pose_list.append([(score + self.scorer.rot_plt) * DOCK_SCORE_K, coords])
        return pose_list


def score_only(rec_mol, lig_mol_list, rec_pred_H=True, lig_pred_H=True,
               use_gauss=True, gscore_params_path=GAUSS_PARAMS_PATH):
    gscore_params_path = gscore_params_path or GAUSS_PARAMS_PATH
    if rec_pred_H:
        rec_mol = protonate_mol(Chem.RemoveHs(Chem.Mol(rec_mol)))
    if lig_pred_H:
        lig_mol_list = [protonate_mol(Chem.RemoveHs(Chem.Mol(lig_mol_list[i]))) for i in range(len(lig_mol_list))]

    score_list = []
    for lig_mol in lig_mol_list:
        scorer = EScorer(rec_mol, lig_mol, DOCKING_PARAMS_PATH, DIST_CUTOFF, PAD_SIZE)
        coords = Chem.RemoveAllHs(Chem.Mol(lig_mol)).GetConformer().GetPositions()
        score = scorer.calc_escore(coords) + scorer.rot_plt
        if use_gauss:
            gauss_scorer = GScorer(rec_mol, lig_mol, gscore_params_path, DIST_CUTOFF, PAD_SIZE)
            score += gauss_scorer.calc_gscore(lig_mol.GetConformer().GetPositions())
        score_list.append(DOCK_SCORE_K * score)

    return score_list
