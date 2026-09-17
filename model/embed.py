import numpy as np
from numba import njit
from rdkit import Chem

from clipdock.atom_type import (EMBED_CENTERS, EMBED_PIS, EMBED_SIGMAS,
                               EMBED_ATOM_TYPE_NUM,
                               type_class, dist_cutoff)
from clipdock.utils.calc import cal_gauss
from clipdock.utils.mol import protonate_mol


@njit(error_model='numpy', boundscheck=False, cache=True)
def radial_basis_embed(coords1, coords2, coords1_type, coords2_type, cutoff, mask):
    n_lig = coords1.shape[0]
    n_rec = coords2.shape[0]
    n_centers = EMBED_CENTERS.shape[0]
    cutoff2 = cutoff * cutoff

    embedding_dim = EMBED_ATOM_TYPE_NUM * n_centers
    embedding = np.zeros((EMBED_ATOM_TYPE_NUM, embedding_dim))
    for i in range(n_lig):
        for j in range(n_rec):
            if mask is not None:
                if i == j or mask[i, j] > 0:
                    continue
            dx = coords1[i, 0] - coords2[j, 0]
            dy = coords1[i, 1] - coords2[j, 1]
            dz = coords1[i, 2] - coords2[j, 2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > cutoff2:
                continue
            d = np.sqrt(d2)

            for k in range(n_centers):
                g_val = EMBED_PIS[k] * cal_gauss(EMBED_SIGMAS[k], EMBED_CENTERS[k], d)
                col_idx = coords2_type[j] * n_centers + k
                embedding[coords1_type[i], col_idx] += g_val
    return embedding


def cpx_embed(rec_mol, lig_mol,
              is_addHs=True, modify_i=None, modify_type='remove'):
    if is_addHs:
        rec_mol = protonate_mol(Chem.Mol(rec_mol))
        lig_mol = protonate_mol(Chem.Mol(lig_mol))

    rec_types = np.ascontiguousarray([type_class.assign_atom_type_id(atom) for atom in rec_mol.GetAtoms()], dtype=np.int32)
    lig_types = np.ascontiguousarray([type_class.assign_atom_type_id(atom) for atom in lig_mol.GetAtoms()], dtype=np.int32)
    rec_coords = np.ascontiguousarray(rec_mol.GetConformer().GetPositions(), dtype=np.float32)
    lig_coords = np.ascontiguousarray(lig_mol.GetConformer().GetPositions(), dtype=np.float32)

    if modify_i is not None:
        if modify_type == 'remove':
            lig_types = np.ascontiguousarray(np.concatenate((lig_types[:modify_i], lig_types[modify_i+1:])))
            lig_coords = np.ascontiguousarray(np.concatenate((lig_coords[:modify_i], lig_coords[modify_i+1:])))
        elif modify_type is not None:
            lig_types[modify_i] = type_class.type_atom_to_id[modify_type]

    return radial_basis_embed(lig_coords, rec_coords, lig_types, rec_types, dist_cutoff, None)


def model_input_prepare(rec_mol, lig_mol_list,
                        is_addHs=True, modify=False, modify_type='remove'):
    if modify:
        is_addHs = False
    x_list = []
    for i, lig_mol in enumerate(lig_mol_list):
        modify_i = i if modify else None
        embedding = cpx_embed(rec_mol, lig_mol, is_addHs=is_addHs, modify_i=modify_i, modify_type=modify_type)
        x_list.append(embedding)

    return np.array(x_list, dtype=np.float32)
