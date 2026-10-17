"""T9-T10: seeded search trajectories and deterministic end-to-end docking."""

import numpy as np
import pytest

import regression_support as support

REFS = ['v100', 'main']


@pytest.mark.parametrize('ref', REFS)
def test_seeded_trajectory(traj_run, ref):
    """Same init conformer, same RNG: trajectories must agree.

    Ligands with rotatable bonds must match bitwise.  Rigid ligands (0 rotatable
    bonds) are allowed a few ULP: the fused objective lets LLVM contract the
    gradient expression into FMA, which changes the last bits of the parameter
    gradient for that branch.
    """
    support.require_distinct(traj_run, ref)
    ref_arrays, _ = support.load_task(traj_run[ref]['dir'], 't9')
    dev_arrays, _ = support.load_task(traj_run['dev']['dir'], 't9')
    checked = {'exact': [], 'ulp': []}
    for key, value in ref_arrays.items():
        if not key.endswith(':scores'):
            continue
        name = key[:-len(':scores')]
        rot = int(dev_arrays[name + ':rot'][0])
        for suffix in (':scores', ':coords'):
            stats = support.diff_stats(ref_arrays[name + suffix], dev_arrays[name + suffix])
            if rot > 0:
                assert stats['equal'], f'{name}{suffix} (rot={rot}) differs: {stats}'
            else:
                assert stats['max_abs'] <= 1e-9, f'{name}{suffix} (rigid) differs: {stats}'
        checked['exact' if rot > 0 else 'ulp'].append(name)
    assert checked['exact'], 'no flexible ligand covered by the trajectory test'
    assert checked['ulp'], 'rigid ligand coverage missing'


@pytest.mark.parametrize('ref', REFS)
def test_minimize_docking_bitwise(fast_run, ref):
    """`dock_func(minimize=True)` is deterministic, so it must match exactly."""
    support.require_distinct(fast_run, ref)
    ref_arrays, ref_info = support.load_task(fast_run[ref]['dir'], 't10')
    dev_arrays, dev_info = support.load_task(fast_run['dev']['dir'], 't10')
    assert ref_arrays, 'reference produced no docking output'
    assert ref_info['docking']['9iwy'].get('error') is None
    for key, value in ref_arrays.items():
        stats = support.diff_stats(value, dev_arrays[key])
        assert stats['equal'], f'{key} differs: {stats}'
    for name, info in ref_info['docking'].items():
        assert info.get('block') == dev_info['docking'][name].get('block'), \
            f'{name}: minimised pose MolBlock differs'


@pytest.mark.parametrize('ref', REFS)
def test_score_only_bitwise(fast_run, ref):
    support.require_distinct(fast_run, ref)
    ref_arrays, ref_info = support.load_task(fast_run[ref]['dir'], 't10')
    dev_arrays, _ = support.load_task(fast_run['dev']['dir'], 't10')
    assert 'score_only' in ref_arrays, ref_info.get('score_only')
    stats = support.diff_stats(ref_arrays['score_only'], dev_arrays['score_only'])
    assert stats['equal'], f'score_only differs: {stats}'
