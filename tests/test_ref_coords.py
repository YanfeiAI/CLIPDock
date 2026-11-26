import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

ETHANOL_ATOMS = [
    (0.0000, 0.0000, 0.0000, 'C'),
    (1.5200, 0.0000, 0.0000, 'C'),
    (2.0100, 1.4100, 0.0000, 'O'),
]
ETHANOL_BONDS = [(1, 2, 1), (2, 3, 1)]
ETHANOL_COORDS = np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0], [2.01, 1.41, 0.0]])

AROMATIC_ATOMS = [
    (0.0000, 1.2000, 0.0000, 'C'),
    (1.1400, 0.3700, 0.0000, 'C'),
    (0.7100, -1.0000, 0.0000, 'C'),
    (-0.7100, -1.0000, 0.0000, 'C'),
    (-1.1400, 0.3700, 0.0000, 'C'),
]
AROMATIC_BONDS = [(1, 2, 4), (2, 3, 4), (3, 4, 4), (4, 5, 4), (5, 1, 4)]

VALENCE_ATOMS = [
    (0.0000, 0.0000, 0.0000, 'N'),
    (1.4000, 0.0000, 0.0000, 'C'),
    (-1.4000, 0.0000, 0.0000, 'C'),
    (0.0000, 1.4000, 0.0000, 'C'),
    (0.0000, -1.4000, 0.0000, 'C'),
    (0.0000, 0.0000, 1.4000, 'C'),
]
VALENCE_BONDS = [(1, 2, 1), (1, 3, 1), (1, 4, 1), (1, 5, 1), (1, 6, 1)]

MOL2_TEXT = """@<TRIPOS>MOLECULE
ref
 2 1 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 C1          1.0000    2.0000    3.0000 C.3       1  LIG         0.0000
      2 O1          4.0000    5.0000    6.0000 O.3       1  LIG         0.0000
@<TRIPOS>BOND
     1    1    2    1
"""


def _molblock(atoms, bonds, title='ref'):
    lines = [title, '  clipdock', '',
             f'{len(atoms):3d}{len(bonds):3d}  0  0  0  0  0  0  0  0999 V2000']
    lines.extend(f'{x:10.4f}{y:10.4f}{z:10.4f} {el:<3s} 0  0  0  0  0  0  0  0  0  0  0  0'
                 for x, y, z, el in atoms)
    lines.extend(f'{i:3d}{j:3d}{t:3d}  0' for i, j, t in bonds)
    lines.append('M  END')
    return '\n'.join(lines) + '\n'


def _write_sdf(path, *records):
    path.write_text(''.join(_molblock(*record) + '$$$$\n' for record in records), encoding='utf-8')
    return path


def _coords_of(atoms):
    return np.array([[x, y, z] for x, y, z, _ in atoms])


def _bad_reference(tmp_path):
    Chem = pytest.importorskip('rdkit.Chem')
    mol = Chem.MolFromMolFile(str(REPO / 'data' / '9iwy_ligand.sdf'), removeHs=False)
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.BondType.AROMATIC)
    for atom in mol.GetAtoms():
        atom.SetIsAromatic(True)
    path = tmp_path / 'bad_ref.sdf'
    path.write_text(Chem.MolToMolBlock(mol, kekulize=False) + '$$$$\n', encoding='utf-8')
    return path


def _dock_args(tmp_path, **overrides):
    fields = dict(receptor=None, ligand=None, smi=None, ref=None,
                  output=str(tmp_path / 'docked.sdf'), skip_extract_pocket=False,
                  pad=8, workers=1, restarts=1, pose_num=1, minimize=False,
                  visualize=False, quiet=True, score_only=False, gscore=None)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _stub_docking(tmp_path, monkeypatch, receptor_name='9iwy_protein.pdb'):
    from clipdock import dock

    receptor = tmp_path / receptor_name
    shutil.copy(REPO / 'data' / receptor_name, receptor)
    seen = {}

    def fake_dock_func(rec_mol, lig_mol, **kwargs):
        seen['rec'] = rec_mol
        seen['lig'] = lig_mol
        return [-1.0], [lig_mol]

    monkeypatch.setattr(dock, 'dock_func', fake_dock_func)
    calls = []
    real_read_coords = dock.read_coords
    monkeypatch.setattr(dock, 'read_coords',
                        lambda path: (calls.append(path), real_read_coords(path))[1])
    return dock, receptor, seen, calls


def test_v2000_coordinates_match_file(tmp_path):
    pytest.importorskip('rdkit.Chem')
    from clipdock.utils.pocket import read_coords

    path = _write_sdf(tmp_path / 'ref.sdf', (ETHANOL_ATOMS, ETHANOL_BONDS))
    np.testing.assert_allclose(read_coords(path), ETHANOL_COORDS, atol=1e-6)


def test_v3000_coordinates_match_file(tmp_path):
    Chem = pytest.importorskip('rdkit.Chem')
    from clipdock.utils.pocket import read_coords

    mol = Chem.MolFromMolFile(str(_write_sdf(tmp_path / 'ref.sdf', (ETHANOL_ATOMS, ETHANOL_BONDS))),
                              removeHs=False)
    v3000 = tmp_path / 'ref_v3000.sdf'
    writer = Chem.SDWriter(str(v3000))
    writer.SetForceV3000(True)
    writer.write(mol)
    writer.close()

    np.testing.assert_allclose(read_coords(v3000), ETHANOL_COORDS, atol=1e-6)


def test_mol2_coordinates_match_file(tmp_path):
    pytest.importorskip('rdkit.Chem')
    from clipdock.utils.pocket import read_coords

    path = tmp_path / 'ref.mol2'
    path.write_text(MOL2_TEXT, encoding='utf-8')
    np.testing.assert_allclose(read_coords(path),
                               np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), atol=1e-6)


def test_explicit_hydrogens_are_excluded(tmp_path):
    pytest.importorskip('rdkit.Chem')
    from clipdock.utils.pocket import read_coords

    atoms = ETHANOL_ATOMS + [(90.0000, 90.0000, 90.0000, 'H')]
    path = _write_sdf(tmp_path / 'ref.sdf', (atoms, ETHANOL_BONDS + [(1, 4, 1)]))
    coords = read_coords(path)
    assert coords.shape == (3, 3)
    np.testing.assert_allclose(coords, ETHANOL_COORDS, atol=1e-6)


def test_kekulization_failure_uses_unsanitized_coordinates(tmp_path):
    Chem = pytest.importorskip('rdkit.Chem')
    from clipdock.utils.pocket import read_coords

    path = _write_sdf(tmp_path / 'ref.sdf', (AROMATIC_ATOMS, AROMATIC_BONDS))
    assert Chem.SDMolSupplier(str(path))[0] is None
    np.testing.assert_allclose(read_coords(path), _coords_of(AROMATIC_ATOMS), atol=1e-6)


def test_invalid_valence_uses_unsanitized_coordinates(tmp_path):
    Chem = pytest.importorskip('rdkit.Chem')
    from clipdock.utils.pocket import read_coords

    path = _write_sdf(tmp_path / 'ref.sdf', (VALENCE_ATOMS, VALENCE_BONDS))
    assert Chem.SDMolSupplier(str(path))[0] is None
    np.testing.assert_allclose(read_coords(path), _coords_of(VALENCE_ATOMS), atol=1e-6)


def test_multi_record_sdf_uses_first_record(tmp_path):
    pytest.importorskip('rdkit.Chem')
    from clipdock.utils.pocket import read_coords

    path = _write_sdf(tmp_path / 'ref.sdf',
                      (ETHANOL_ATOMS, ETHANOL_BONDS),
                      (VALENCE_ATOMS, VALENCE_BONDS))
    np.testing.assert_allclose(read_coords(path), ETHANOL_COORDS, atol=1e-6)


def test_truncated_sdf_raises(tmp_path):
    pytest.importorskip('rdkit.Chem')
    from clipdock.utils.pocket import read_coords

    block = _molblock(ETHANOL_ATOMS, ETHANOL_BONDS)
    path = tmp_path / 'truncated.sdf'
    path.write_text(block[:len(block) // 2], encoding='utf-8')
    with pytest.raises(ValueError, match='reference coordinates'):
        read_coords(path)


def test_empty_sdf_raises(tmp_path):
    pytest.importorskip('rdkit.Chem')
    from clipdock.utils.pocket import read_coords

    path = tmp_path / 'empty.sdf'
    path.write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='reference coordinates'):
        read_coords(path)


def test_nonfinite_coordinates_raise(tmp_path, monkeypatch):
    Chem = pytest.importorskip('rdkit.Chem')
    from clipdock.utils import pocket

    mol = Chem.MolFromMolBlock(_molblock(ETHANOL_ATOMS, ETHANOL_BONDS), removeHs=False)
    mol.GetConformer().SetAtomPosition(0, (float('nan'), 0.0, 0.0))
    monkeypatch.setattr(pocket, 'read_mol', lambda *args, **kwargs: Chem.Mol(mol))
    with pytest.raises(ValueError, match='reference coordinates'):
        pocket.read_coords(tmp_path / 'ref.sdf')


def test_same_named_mol2_is_not_substituted(tmp_path):
    pytest.importorskip('rdkit.Chem')
    from clipdock.utils.pocket import read_coords

    (tmp_path / 'ref.sdf').write_text('', encoding='utf-8')
    (tmp_path / 'ref.mol2').write_text(MOL2_TEXT, encoding='utf-8')
    with pytest.raises(ValueError, match='reference coordinates'):
        read_coords(tmp_path / 'ref.sdf')


def test_dock_shares_one_reference_for_pocket_and_centring(tmp_path, monkeypatch):
    pytest.importorskip('rdkit.Chem')
    from clipdock.utils.mol import read_mol
    from clipdock.utils.pocket import extract_pocket_from_coords, read_coords

    dock, receptor, seen, calls = _stub_docking(tmp_path, monkeypatch)
    ref = _bad_reference(tmp_path)
    dock.main(_dock_args(tmp_path, receptor=str(receptor), ref=str(ref),
                         ligand=str(REPO / 'data' / '9iwy_ligand.sdf')))

    assert len(calls) == 1
    ref_coords = read_coords(ref)
    np.testing.assert_allclose(seen['lig'].GetConformer().GetPositions().mean(axis=0),
                               ref_coords.mean(axis=0), atol=1e-6)

    pocket = receptor.parent / (receptor.stem + '_pocket.pdb')
    expected = tmp_path / 'expected_pocket.pdb'
    extract_pocket_from_coords(ref_coords, str(receptor), str(expected), remove_water=True)
    assert pocket.read_text() == expected.read_text()
    assert seen['rec'].GetNumAtoms() == read_mol(pocket).GetNumAtoms()
    assert seen['rec'].GetNumAtoms() < read_mol(receptor).GetNumAtoms()


def test_dock_skip_extract_pocket_still_centres_on_reference(tmp_path, monkeypatch):
    pytest.importorskip('rdkit.Chem')
    from clipdock.utils.mol import read_mol
    from clipdock.utils.pocket import read_coords

    dock, receptor, seen, calls = _stub_docking(tmp_path, monkeypatch,
                                                receptor_name='9iwy_protein_pocket.pdb')
    ref = _bad_reference(tmp_path)
    dock.main(_dock_args(tmp_path, receptor=str(receptor), ref=str(ref), skip_extract_pocket=True,
                         ligand=str(REPO / 'data' / '9iwy_ligand.sdf')))

    assert len(calls) == 1
    assert not (tmp_path / '9iwy_protein_pocket_pocket.pdb').exists()
    np.testing.assert_allclose(seen['lig'].GetConformer().GetPositions().mean(axis=0),
                               read_coords(ref).mean(axis=0), atol=1e-6)
    assert seen['rec'].GetNumAtoms() == read_mol(receptor).GetNumAtoms()


def test_dock_smiles_centres_on_reference(tmp_path, monkeypatch):
    pytest.importorskip('rdkit.Chem')
    from clipdock.utils.pocket import read_coords

    dock, receptor, seen, calls = _stub_docking(tmp_path, monkeypatch,
                                                receptor_name='9iwy_protein_pocket.pdb')
    ref = _bad_reference(tmp_path)
    dock.main(_dock_args(tmp_path, receptor=str(receptor), ref=str(ref), smi='CCO',
                         skip_extract_pocket=True))

    assert len(calls) == 1
    np.testing.assert_allclose(seen['lig'].GetConformer().GetPositions().mean(axis=0),
                               read_coords(ref).mean(axis=0), atol=1e-6)


def test_dock_still_rejects_invalid_ligand(tmp_path, monkeypatch):
    pytest.importorskip('rdkit.Chem')
    from clipdock import dock

    _, receptor, _, _ = _stub_docking(tmp_path, monkeypatch,
                                      receptor_name='9iwy_protein_pocket.pdb')
    ligand = _write_sdf(tmp_path / 'ligand.sdf', (VALENCE_ATOMS, VALENCE_BONDS))
    args = _dock_args(tmp_path, receptor=str(receptor), ligand=str(ligand),
                      ref=str(REPO / 'data' / '9iwy_ligand.sdf'), skip_extract_pocket=True)
    with pytest.raises(ValueError, match='Failed to read Ligand'):
        dock.main(args)
