"""T1-T3: charges, atom types and protonation against the real v1.0.0 source."""

import numpy as np
import pytest

import regression_support as support

REFS = ['v100', 'main']


def _pair(fast_run, ref, task):
    support.require_distinct(fast_run, ref)
    a, ai = support.load_task(fast_run[ref]['dir'], task)
    b, bi = support.load_task(fast_run['dev']['dir'], task)
    return a, ai, b, bi


def _assert_exact(ref_arrays, dev_arrays, skip_suffix=(':batch',)):
    assert ref_arrays, 'reference produced no arrays'
    for key in ref_arrays:
        assert key in dev_arrays, f'{key} missing from dev output'
        stats = support.diff_stats(ref_arrays[key], dev_arrays[key])
        assert stats['equal'], f'{key} differs: {stats}'


@pytest.mark.parametrize('ref', REFS)
def test_atom_charges_identical(fast_run, ref):
    a, _, b, _ = _pair(fast_run, ref, 't1')
    dev_only = {k for k in b if k.endswith(':batch')}
    _assert_exact(a, {k: v for k, v in b.items() if k not in dev_only})
    assert len(a) >= 4 * 20, f'expected ligand+pocket triples, got {len(a)}'


def test_batch_charge_helper_matches_per_atom(fast_run):
    _, info = support.load_task(fast_run['dev']['dir'], 't1')
    flags = info.get('batch_matches_per_atom') or {}
    assert flags, 'no batch comparison recorded'
    bad = {k: v for k, v in flags.items() if not v}
    assert not bad, f'calc_atom_charges disagrees for {bad}'


@pytest.mark.parametrize('ref', REFS)
def test_atom_types_identical(fast_run, ref):
    a, ainfo, b, binfo = _pair(fast_run, ref, 't2')
    _assert_exact(a, b)
    assert ainfo['dtypes'] == binfo['dtypes'], 'atom-type array dtype changed'
    assert len(a) == len(ainfo['dtypes'])


@pytest.mark.parametrize('ref', REFS)
def test_protonation_identical(fast_run, ref):
    a, ainfo, b, binfo = _pair(fast_run, ref, 't3')
    assert ainfo['molecules'] == binfo['molecules'], 'protonation changed atom count/bonds/SMILES'
    _assert_exact(a, b)


def test_protonation_covers_functional_groups(fast_run):
    _, info = support.load_task(fast_run['dev']['dir'], 't3')
    names = {k.split(':')[0] for k in info['molecules']}
    required = {'pyridine', 'imidazole', 'guanidine', 'carboxylate', 'phosphate',
                'sulfonate', 'quaternary_n', 'chloro', 'iodo', 'nitro', '9iwy'}
    missing = required - names
    assert not missing, f'missing chemical coverage: {missing}'
    assert all('error' not in v for v in info['molecules'].values())
