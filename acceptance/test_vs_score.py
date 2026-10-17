"""T11-T12: VS-score graph conversion, CPU inference and receptor-embedding cache."""

import json
import os

import numpy as np
import pytest

import regression_support as support

pytest.importorskip('torch')
pytest.importorskip('pytorch_lightning')

REFS = ['v100', 'main']


@pytest.mark.parametrize('ref', REFS)
def test_graph_conversion_identical(vs_run, ref):
    support.require_distinct(vs_run, ref)
    ref_arrays, ref_info = support.load_task(vs_run[ref]['dir'], 't11')
    dev_arrays, _ = support.load_task(vs_run['dev']['dir'], 't11')
    assert not ref_info.get('error'), ref_info['error']
    graph_keys = [k for k in ref_arrays if not k.endswith(':scores')]
    assert graph_keys, 'no graph arrays compared'
    for key in graph_keys:
        stats = support.diff_stats(ref_arrays[key], dev_arrays[key])
        assert stats['equal'], f'{key} differs: {stats}'


@pytest.mark.parametrize('ref', REFS)
def test_cpu_scores_bitwise(vs_run, ref):
    """CPU inference is deterministic, so scores must match bitwise."""
    support.require_distinct(vs_run, ref)
    ref_arrays, _ = support.load_task(vs_run[ref]['dir'], 't11')
    dev_arrays, _ = support.load_task(vs_run['dev']['dir'], 't11')
    assert 'scores' in ref_arrays
    assert np.isfinite(ref_arrays['scores']).all()
    stats = support.diff_stats(ref_arrays['scores'], dev_arrays['scores'])
    assert stats['equal'], f'VS-score CPU scores differ: {stats}'


def test_receptor_cache_matches_uncached(vs_run):
    """The cache must be transparent: identical scores with and without it."""
    import torch
    from rdkit import Chem
    from model import graph_score as gs
    from model.graph import ligand_to_graph, protein_to_graph
    from model.graph_utils import collate_graph_attributes
    from model.train_vs import VSTrainer

    torch.set_num_threads(1)
    ckpt = os.environ.get('CLIPDOCK_VS_CKPT')
    with open(support.FIXTURE_ROOT / 'molecules.json') as f:
        entries = json.load(f)['entries']
    recs = [e for e in entries if e['name'] == '9iwy']
    assert recs
    pocket_mol = Chem.MolFromPDBFile(recs[0]['pocket'])
    g_rec = protein_to_graph(pocket_mol)
    lig_graphs = []
    for e in entries[:14]:
        mol = Chem.SDMolSupplier(e['ligand'], removeHs=False)[0]
        g = ligand_to_graph(mol)
        if g is not None:
            lig_graphs.append(g)
    assert len(lig_graphs) >= 13

    device = torch.device('cpu')
    model = VSTrainer.load_from_checkpoint(ckpt, map_location=device)
    model.eval()

    def run(bs, cache):
        out = []
        for start in range(0, len(lig_graphs), bs):
            chunk = lig_graphs[start:start + bs]
            data = [{'prot': g_rec, 'lig': g} for g in chunk]
            batch = (collate_graph_attributes(data, 'prot'), collate_graph_attributes(data, 'lig'))
            out.extend(gs.compute_mdn_scores_batch(model, batch, device, protein_cache=cache))
        return np.asarray(out, dtype=np.float64)

    for bs in (1, 4, 13):
        cached = run(bs, gs.ProteinEmbeddingCache())
        plain = run(bs, None)
        assert np.array_equal(cached, plain), f'cache changed scores at batch size {bs}'


def test_cache_rejects_other_pocket(vs_run):
    from model import graph_score as gs

    cache = gs.ProteinEmbeddingCache(pocket_key=('a.pdb', 1))
    try:
        cache.get(None, None, None, None, None, False, pocket_key=('b.pdb', 2))
    except (ValueError, TypeError, AttributeError) as exc:
        assert isinstance(exc, ValueError), f'unexpected error: {exc!r}'
    else:  # pragma: no cover
        raise AssertionError('cache accepted a different pocket key')
