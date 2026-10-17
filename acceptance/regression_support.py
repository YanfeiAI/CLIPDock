"""Shared plumbing for the cross-version regression tests.

Reference trees are real Git worktrees of v1.0.0 and main (or directories given
by ``CLIPDOCK_REF_V100`` / ``CLIPDOCK_REF_MAIN``), so every comparison runs the
actual historical source in its own subprocess.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TREE_ROOT = REPO_ROOT / 'tmp' / 'regression_trees'
FIXTURE_ROOT = REPO_ROOT / 'tmp' / 'regression_fixtures'
DRIVER = REPO_ROOT / 'acceptance' / 'regression_driver.py'
NUMBA_CACHE = REPO_ROOT / 'tmp' / 'regression_numba_cache'

# ligand-only fixtures cover the functional groups that drive charges/typing
SMILES_FIXTURES = [
    ('pyridine', 'c1ccncc1'),
    ('imidazole', 'c1cnc[nH]1'),
    ('guanidine', 'NC(=N)N'),
    ('amidine', 'CC(=N)N'),
    ('carboxylate', 'CC(=O)[O-]'),
    ('carboxylic_acid', 'CC(=O)O'),
    ('phosphate', 'OP(=O)(O)OC'),
    ('sulfonate', 'CS(=O)(=O)[O-]'),
    ('quaternary_n', 'C[N+](C)(C)C'),
    ('primary_amine', 'CCN'),
    ('chloro', 'c1ccc(Cl)cc1'),
    ('bromo', 'c1ccc(Br)cc1'),
    ('iodo', 'c1ccc(I)cc1'),
    ('thiol', 'CCS'),
    ('sulfonamide', 'CS(=O)(=O)N'),
    ('nitro', 'c1ccc([N+](=O)[O-])cc1'),
    ('amide', 'CC(=O)NC'),
    ('nitrile', 'CC#N'),
    ('benzene_sym', 'c1ccccc1'),
    ('para_xylyl', 'Cc1ccc(C)cc1'),
]


def _git(*args, cwd=REPO_ROOT):
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def ensure_tree(ref):
    """Return the source root of a reference version, creating a worktree if needed."""
    env = {'v100': 'CLIPDOCK_REF_V100', 'main': 'CLIPDOCK_REF_MAIN'}.get(ref)
    if env and os.environ.get(env):
        path = Path(os.environ[env]).resolve()
        assert (path / 'clipdock' / 'scoring.py').is_file(), f'{env}={path} is not a CLIPDock tree'
        return path
    if ref == 'dev':
        return REPO_ROOT
    path = TREE_ROOT / ref
    if not (path / 'clipdock' / 'scoring.py').is_file():
        TREE_ROOT.mkdir(parents=True, exist_ok=True)
        rev = {'v100': 'v1.0.0', 'main': 'main'}[ref]
        _git('worktree', 'add', '--detach', '-f', str(path), rev)
    return path


def tree_commit(tree):
    try:
        return _git('rev-parse', 'HEAD', cwd=tree)
    except Exception:
        return None


def require_distinct(run, ref):
    """Skip a comparison whose reference tree is the same commit as the tree under test.

    ``main`` is the development baseline, so after a release merge it *is* the
    released commit and comparing it with ``dev`` would just compare the tree with
    itself.  Skipping keeps that visible instead of passing on a tautology.
    """
    import pytest

    same = run[ref].get('commit')
    if same and same == run['dev'].get('commit'):
        pytest.skip(f'{ref} and dev are both {same[:8]}; nothing to compare')


def build_molecules(limit=None, extra_pb=0):
    """Freeze the molecule list used by every version.

    Uses the bundled 9iwy example (present in every checkout), a fixed set of
    functional-group SMILES embedded once with a fixed ETKDG seed, and - when a
    PoseBusters manifest is available - real complexes from it.
    """
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    data = REPO_ROOT / 'data'
    entries = [{'name': '9iwy', 'ligand': str(data / '9iwy_ligand.sdf'),
                'receptor': str(data / '9iwy_protein.pdb'), 'source': 'bundled'}]

    from rdkit import Chem
    from rdkit.Chem import AllChem
    for name, smi in SMILES_FIXTURES:
        sdf = FIXTURE_ROOT / f'{name}.sdf'
        if not sdf.is_file():
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xC11D  # fixed fixture seed
            params.useSmallRingTorsions = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                AllChem.EmbedMolecule(mol, randomSeed=0xC11D, useRandomCoords=True)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
            mol.SetProp('_Name', name)
            with Chem.SDWriter(str(sdf)) as w:
                w.write(mol)
        entries.append({'name': name, 'ligand': str(sdf),
                        'receptor': str(data / '9iwy_protein.pdb'), 'source': 'smiles'})

    manifest = os.environ.get('CLIPDOCK_PB_MANIFEST')
    if manifest and Path(manifest).is_file() and extra_pb:
        import csv
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        step = max(1, len(rows) // extra_pb)
        for r in rows[::step][:extra_pb]:
            entries.append({'name': r['item'], 'ligand': r['ligand'], 'receptor': r['receptor'],
                            'source': 'posebusters'})
    if limit:
        entries = entries[:limit]

    payload = {'version': 1, 'entries': entries}
    path = FIXTURE_ROOT / 'molecules.json'
    path.write_text(json.dumps(payload, indent=1))
    return path


def prepare_pockets(molecules_json, out_dir, tree=None):
    """Extract a pocket for entries that only reference a full receptor."""
    with open(molecules_json) as f:
        payload = json.load(f)
    if not any('pocket' not in e and 'receptor' in e for e in payload['entries']):
        return molecules_json
    tree = tree or ensure_tree('dev')
    out_dir = Path(out_dir)
    run_driver(tree, molecules_json, ['t0'], out_dir)
    with open(out_dir / 't0.json') as f:
        mapping = json.load(f)
    for e in payload['entries']:
        if 'pocket' not in e and e['name'] in mapping:
            e['pocket'] = mapping[e['name']]
    Path(molecules_json).write_text(json.dumps(payload, indent=1))
    return molecules_json


def run_driver(tree, molecules_json, tasks, out_dir, extra_env=None):
    """Execute the driver inside ``tree`` and return its meta.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(tree)
    env.setdefault('NUMBA_NUM_THREADS', '1')
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env['CLIPDOCK_DATA_ROOT'] = str(REPO_ROOT / 'tmp' / 'regression_data')
    # numba's on-disk cache can serve a kernel built from an earlier revision of
    # the same file (the source stamp is only mtime+size), so every run gets a
    # private cache and no leftover artifacts in the tree.
    for package in ('clipdock', 'model'):
        shutil.rmtree(Path(tree) / package / '__pycache__', ignore_errors=True)
    cache_dir = Path(tempfile.mkdtemp(prefix='numba_', dir=str(NUMBA_CACHE)))
    env['NUMBA_CACHE_DIR'] = str(cache_dir)
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, str(DRIVER), '--molecules', str(molecules_json),
           '--task', *tasks, '--out', str(out_dir)]
    proc = subprocess.run(cmd, env=env, cwd=str(tree), capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f'driver failed in {tree}:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}')
    with open(out_dir / 'meta.json') as f:
        return json.load(f)


def load_task(out_dir, task):
    with np.load(Path(out_dir) / f'{task}.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    with open(Path(out_dir) / f'{task}.json') as f:
        info = json.load(f)
    return arrays, info


def diff_stats(a, b):
    """Exact/floating comparison summary for two arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {'shape': (a.shape, b.shape), 'equal': False, 'max_abs': float('inf'), 'ulps': None}
    if a.dtype.kind in 'iu' or a.dtype.kind == 'b':
        return {'equal': bool(np.array_equal(a, b)),
                'max_abs': 0.0 if np.array_equal(a, b) else float('inf'), 'ulps': None}
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    same = af == bf
    if np.all(same):
        return {'equal': True, 'max_abs': 0.0, 'max_rel': 0.0, 'ulps': 0}
    d = np.abs(af - bf)
    denom = np.maximum(np.abs(af), np.abs(bf))
    denom[denom == 0] = 1.0
    scale = np.abs(np.spacing(np.where(np.isfinite(af) & (af != 0), af, 1.0)))
    ulps = np.where(same, 0, d / np.maximum(scale, 1e-300))
    return {'equal': False,
            'max_abs': float(np.nanmax(d)),
            'max_rel': float(np.nanmax(d / denom)),
            'ulps': float(np.nanmax(ulps)),
            'n_diff': int(np.count_nonzero(~same)),
            'n_total': int(af.size)}


def clean_regression_dirs():
    for d in (TREE_ROOT, FIXTURE_ROOT):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
