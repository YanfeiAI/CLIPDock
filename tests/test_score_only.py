"""CLI wiring for --score_only; no scoring kernels or reference tree needed."""

from types import SimpleNamespace

import pytest


def _write_sdf(path):
    Chem = pytest.importorskip('rdkit.Chem')
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)


def _stub_dock(monkeypatch):
    from clipdock import dock

    calls = {'score': [], 'visual': []}

    def fake_score_only(rec_mol, mol_list, **kwargs):
        calls['score'].append(kwargs)
        return [0.0] * len(mol_list)

    def fake_visual(*args, **kwargs):
        calls['visual'].append(kwargs)

    monkeypatch.setattr(dock, 'read_mol', lambda *a, **k: object())
    monkeypatch.setattr(dock, 'save_docked_mol', lambda *a, **k: None)
    monkeypatch.setattr(dock, 'score_only', fake_score_only)
    monkeypatch.setattr(dock, 'analyse_scorer_atom_contrib', fake_visual)
    return dock, calls


def _args(tmp_path, visualize):
    return SimpleNamespace(
        ligand=str(tmp_path / 'input.sdf'),
        output=str(tmp_path / 'output.sdf'),
        receptor='receptor.pdb',
        ref=None,
        visualize=visualize,
        gscore='custom_gscore.csv',
    )


def test_score_only_protonates_ligand(tmp_path, monkeypatch):
    pytest.importorskip('rdkit.Chem')
    dock, calls = _stub_dock(monkeypatch)
    args = _args(tmp_path, visualize=False)
    _write_sdf(args.ligand)

    dock.score_input_file(args)

    assert calls['score'][0]['lig_pred_H'] is True
    assert calls['score'][0]['gscore_params_path'] == 'custom_gscore.csv'


def test_score_only_visualize_protonates_ligand(tmp_path, monkeypatch):
    pytest.importorskip('rdkit.Chem')
    dock, calls = _stub_dock(monkeypatch)
    args = _args(tmp_path, visualize=True)
    _write_sdf(args.ligand)

    dock.score_input_file(args)

    assert calls['score'][0]['lig_pred_H'] is True
    assert calls['visual'][0]['lig_pred_H'] is True
    assert calls['visual'][0]['gscore_params_path'] == 'custom_gscore.csv'
