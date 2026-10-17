"""Session fixtures: reference worktrees, frozen inputs, driver runs.

Run with ``pytest acceptance/``; a bare ``pytest`` only covers ``tests/``.
"""

import os
import sys
from pathlib import Path

import pytest

import regression_support as support

# tests run from the repository root; make the development tree importable
sys.path.insert(0, str(support.REPO_ROOT))

TASKS_FAST = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't10']


def _prepare_molecules():
    limit = int(os.environ.get('CLIPDOCK_REGRESSION_LIMIT', '0')) or None
    extra_pb = int(os.environ.get('CLIPDOCK_REGRESSION_PB', '0'))
    path = support.build_molecules(limit=limit, extra_pb=extra_pb)
    pocket_dir = support.FIXTURE_ROOT / 'pockets'
    support.prepare_pockets(path, pocket_dir)
    return path


@pytest.fixture(scope='session', autouse=True)
def _fresh_numba_cache(trees):
    """Give the whole session a private numba cache.

    numba's on-disk cache does not always notice that a kernel changed when a
    source file is replaced, and the reference worktrees keep their own caches;
    both would let a stale build silently stand in for the current source.
    """
    import shutil
    shutil.rmtree(support.NUMBA_CACHE, ignore_errors=True)
    support.NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ['NUMBA_CACHE_DIR'] = str(support.NUMBA_CACHE / 'inproc')
    for tree in trees.values():
        for package in ('clipdock', 'model'):
            shutil.rmtree(Path(tree) / package / '__pycache__', ignore_errors=True)


@pytest.fixture(scope='session')
def molecules():
    return _prepare_molecules()


@pytest.fixture(scope='session')
def trees():
    return {name: support.ensure_tree(name) for name in ('v100', 'main', 'dev')}


@pytest.fixture(scope='session')
def run_root():
    root = support.REPO_ROOT / 'tmp' / 'regression_runs'
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope='session')
def fast_run(trees, molecules, run_root):
    """Driver output for T1-T8/T10 on every version."""
    out = {}
    for name, tree in trees.items():
        target = run_root / f'fast_{name}'
        meta = support.run_driver(tree, molecules, TASKS_FAST, target)
        out[name] = {'dir': target, 'meta': meta, 'tree': tree,
                     'commit': support.tree_commit(tree)}
    return out


@pytest.fixture(scope='session')
def traj_run(trees, molecules, run_root):
    out = {}
    for name, tree in trees.items():
        target = run_root / f'traj_{name}'
        meta = support.run_driver(tree, molecules, ['t9'], target)
        out[name] = {'dir': target, 'meta': meta, 'tree': tree,
                     'commit': support.tree_commit(tree)}
    return out


@pytest.fixture(scope='session')
def vs_run(trees, molecules, run_root):
    ckpt = os.environ.get('CLIPDOCK_VS_CKPT')
    if not ckpt or not Path(ckpt).is_file():
        pytest.skip('SKIP: external assets missing (CLIPDOCK_VS_CKPT)')
    out = {}
    for name, tree in trees.items():
        target = run_root / f'vs_{name}'
        meta = support.run_driver(tree, molecules, ['t11'], target,
                                  extra_env={'CLIPDOCK_VS_CKPT': ckpt})
        out[name] = {'dir': target, 'meta': meta, 'tree': tree,
                     'commit': support.tree_commit(tree)}
    return out
