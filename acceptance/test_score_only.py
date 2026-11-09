"""--score_only must reproduce the docking score for a frozen pose."""

from pathlib import Path

import pytest


def test_score_only_matches_docking():
    Chem = pytest.importorskip('rdkit.Chem')
    from clipdock.dock import dock_func
    from clipdock.engine import score_only
    from clipdock.utils.mol import read_mol

    data = Path(__file__).resolve().parents[1] / 'data'
    rec = read_mol(str(data / '9iwy_protein_pocket.pdb'))
    lig = read_mol(str(data / '9iwy_ligand.sdf'))

    scores, mols = dock_func(rec, lig, rec_pred_H=True, lig_pred_H=True,
                             minimize=True, restarts=1, workers=1,
                             pose_num=1, output=None)
    pose = mols[0]
    assert pose.GetNumAtoms() == pose.GetNumHeavyAtoms()

    # Re-protonating a heavy-atom pose reproduces the docking G-score H set.
    assert score_only(rec, [Chem.Mol(pose)], lig_pred_H=True)[0] == scores[0]
    assert score_only(rec, [Chem.Mol(pose)], lig_pred_H=False)[0] != scores[0]
