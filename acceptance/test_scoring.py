"""T4-T8: scores, gradients, transforms, pose filtering and scorer reuse."""

import json

import numpy as np
import pytest

import regression_support as support

REFS = ['v100', 'main']

# The fused objective lets LLVM contract the gradient expression into FMA, which
# moves a few gradient components by ~1e-14 relative (measured max: abs 1.8e-15,
# rel 1.4e-14).  The task book allows 1e-7 absolute / 1e-6 relative for
# gradients; these bounds sit far below that but above the measured ULP noise.
GRAD_ABS = 1e-10
GRAD_REL = 1e-12


def _both(fast_run, ref, task):
    support.require_distinct(fast_run, ref)
    ref_arrays, ref_info = support.load_task(fast_run[ref]['dir'], task)
    dev_arrays, dev_info = support.load_task(fast_run['dev']['dir'], task)
    assert ref_arrays, f'{task}: reference produced no arrays'
    return ref_arrays, ref_info, dev_arrays, dev_info


@pytest.mark.parametrize('ref', REFS)
def test_escore_gscore_and_atom_gradients_bitwise(fast_run, ref):
    ref_arrays, _, dev_arrays, _ = _both(fast_run, ref, 't4')
    for key in ref_arrays:
        stats = support.diff_stats(ref_arrays[key], dev_arrays[key])
        assert stats['equal'], f'{key} differs: {stats}'


@pytest.mark.parametrize('ref', REFS)
def test_parameter_gradient(fast_run, ref):
    """Objective value must match bitwise; gradients may differ by a few ULP."""
    ref_arrays, _, dev_arrays, _ = _both(fast_run, ref, 't5')
    for key, value in ref_arrays.items():
        stats = support.diff_stats(value, dev_arrays[key])
        if key.endswith(':f'):
            assert stats['equal'], f'objective value differs: {key} {stats}'
        else:
            assert stats['max_abs'] <= GRAD_ABS and stats['max_rel'] <= GRAD_REL, \
                f'{key} gradient differs beyond ULP noise: {stats}'


@pytest.mark.parametrize('ref', REFS)
def test_rotatable_bonds_and_transforms(fast_run, ref):
    ref_arrays, _, dev_arrays, _ = _both(fast_run, ref, 't6')
    for key, value in ref_arrays.items():
        stats = support.diff_stats(value, dev_arrays[key])
        if key.endswith((':indices', ':offsets', ':pairs')):
            assert stats['equal'], f'rotatable-bond decomposition differs: {key} {stats}'
        elif key.endswith(':pgrad'):
            assert stats['max_abs'] <= GRAD_ABS and stats['max_rel'] <= GRAD_REL, \
                f'{key} projection differs beyond ULP noise: {stats}'
        else:
            assert stats['equal'], f'{key} differs: {stats}'


@pytest.mark.parametrize('ref', REFS)
def test_rmsd_and_pose_filtering(fast_run, ref):
    ref_arrays, ref_info, dev_arrays, dev_info = _both(fast_run, ref, 't7')
    for key, value in ref_arrays.items():
        stats = support.diff_stats(value, dev_arrays[key])
        assert stats['equal'], f'{key} differs: {stats}'
    assert ref_info == dev_info, 'kept poses, MolBlocks or atom counts differ'


@pytest.mark.parametrize('ref', REFS)
def test_scorer_reuse_has_no_state_leak(fast_run, ref):
    ref_arrays, _, dev_arrays, _ = _both(fast_run, ref, 't8')
    for key, arr in dev_arrays.items():
        if key.endswith(':repeat'):
            assert np.array_equal(arr[0], arr[1]) and np.array_equal(arr[0], arr[2]), \
                f'{key}: repeated construction changed scores'
    for key, value in ref_arrays.items():
        stats = support.diff_stats(value, dev_arrays[key])
        assert stats['equal'], f'{key} differs: {stats}'


def test_finite_difference_gradient(molecules):
    """Local sanity check of the analytic parameter gradient (no reference tree)."""
    from rdkit import Chem

    from clipdock.engine import PoseOptimizer
    from clipdock.utils.constant import DOCKING_PARAMS_PATH, GAUSS_PARAMS_PATH
    from clipdock.utils.mol import protonate_mol, read_mol

    entries = json.loads(open(molecules).read())['entries']
    entry = next(e for e in entries if e['name'] == 'imidazole')
    rec = protonate_mol(Chem.RemoveHs(Chem.Mol(read_mol(entry['pocket']))))
    lig = protonate_mol(Chem.RemoveHs(Chem.Mol(read_mol(entry['ligand']))))
    opt = PoseOptimizer(rec, lig, DOCKING_PARAMS_PATH, 1, 11, 32, 8.0, 8.0,
                        gscore_params_path=GAUSS_PARAMS_PATH)
    x = opt.random_init_x()
    _, grad = opt.objective_with_grad(x)
    grad = np.array(grad, copy=True)

    eps = 1e-6
    fd = np.zeros_like(grad)
    for i in range(len(x)):
        plus, minus = x.copy(), x.copy()
        plus[i] += eps
        minus[i] -= eps
        fd[i] = (opt.objective_with_grad(plus)[0] - opt.objective_with_grad(minus)[0]) / (2 * eps)

    mask = np.abs(fd) > 1e-3
    assert mask.any()
    rel = np.abs(grad[mask] - fd[mask]) / np.abs(fd[mask])
    assert np.median(rel) < 1e-4, f'gradient does not match finite differences: {np.median(rel)}'
