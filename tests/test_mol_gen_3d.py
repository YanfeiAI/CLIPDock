import pytest

SIMPLE_SMILES = ['CCO', 'c1ccccc1', 'C[C@H](N)C(=O)O', 'C1CC2CCC1C2']
BRIDGEHEAD_SMILES = 'N[C@]12C[C@](N)(C1)C2'


def assert_valid_3d(mol, heavy_atoms, with_hydrogens=True):
    assert mol is not None
    assert mol.GetNumConformers() == 1
    assert mol.GetConformer().Is3D()
    assert mol.GetNumHeavyAtoms() == heavy_atoms
    if with_hydrogens:
        assert mol.GetNumAtoms() > heavy_atoms
    else:
        assert mol.GetNumAtoms() == heavy_atoms


@pytest.mark.parametrize('smi', SIMPLE_SMILES)
def test_mol_gen_3d_produces_3d_conformer(smi):
    Chem = pytest.importorskip('rdkit.Chem')
    from clipdock.utils.mol import mol_gen_3d

    mol = mol_gen_3d(Chem.MolFromSmiles(smi))
    assert_valid_3d(mol, Chem.MolFromSmiles(smi).GetNumHeavyAtoms())


def test_mol_gen_3d_recovers_bridgehead_chirality():
    Chem = pytest.importorskip('rdkit.Chem')
    from rdkit.Chem import AllChem
    from clipdock.utils.mol import mol_gen_3d

    plain = Chem.AddHs(Chem.MolFromSmiles(BRIDGEHEAD_SMILES))
    params = AllChem.ETKDGv3()
    params.randomSeed = 1
    AllChem.EmbedMolecule(plain, params)
    assert plain.GetNumConformers() == 0

    mol = mol_gen_3d(Chem.MolFromSmiles(BRIDGEHEAD_SMILES))
    assert_valid_3d(mol, Chem.MolFromSmiles(BRIDGEHEAD_SMILES).GetNumHeavyAtoms())


def test_mol_gen_3d_fallback_stages_are_progressive(monkeypatch):
    Chem = pytest.importorskip('rdkit.Chem')
    from rdkit.Chem import AllChem
    from clipdock.utils import mol as mol_utils

    real_embed = AllChem.EmbedMolecule
    seen = []

    def fake_embed(m, params):
        seen.append((params.randomSeed, params.useRandomCoords, params.enforceChirality,
                     params.maxIterations))
        if params.enforceChirality:
            return -1
        return real_embed(m, params)

    monkeypatch.setattr(mol_utils.AllChem, 'EmbedMolecule', fake_embed)
    mol = mol_utils.mol_gen_3d(Chem.MolFromSmiles(BRIDGEHEAD_SMILES))
    assert_valid_3d(mol, Chem.MolFromSmiles(BRIDGEHEAD_SMILES).GetNumHeavyAtoms())

    assert len(seen) == 3
    assert len({s[0] for s in seen}) == 1
    assert [(s[1], s[2], s[3]) for s in seen] == [
        (False, True, 0),
        (True, True, mol_utils.RETRY_MAX_ITERATIONS),
        (True, False, mol_utils.RETRY_MAX_ITERATIONS),
    ]


def test_mol_gen_3d_returns_none_when_all_stages_fail(monkeypatch):
    Chem = pytest.importorskip('rdkit.Chem')
    from clipdock.utils import mol as mol_utils

    monkeypatch.setattr(mol_utils.AllChem, 'EmbedMolecule', lambda m, p: -1)
    assert mol_utils.mol_gen_3d(Chem.MolFromSmiles('CCO')) is None


def test_mol_gen_3d_returns_none_for_none_input():
    pytest.importorskip('rdkit.Chem')
    from clipdock.utils.mol import mol_gen_3d

    assert mol_gen_3d(None) is None


def test_mol_gen_3d_without_added_hydrogens():
    Chem = pytest.importorskip('rdkit.Chem')
    from clipdock.utils.mol import mol_gen_3d

    smi = 'C[C@H](N)C(=O)O'
    mol = mol_gen_3d(Chem.MolFromSmiles(smi), is_addHs=False)
    assert_valid_3d(mol, Chem.MolFromSmiles(smi).GetNumHeavyAtoms(), with_hydrogens=False)


def test_mol_gen_3d_preserves_stereochemistry():
    Chem = pytest.importorskip('rdkit.Chem')
    from clipdock.utils.mol import mol_gen_3d

    smi = 'C[C@H](N)C(=O)O'
    mol = mol_gen_3d(Chem.MolFromSmiles(smi))
    assert Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False) == \
        Chem.FindMolChiralCenters(Chem.MolFromSmiles(smi), includeUnassigned=True,
                                  useLegacyImplementation=False)


def test_stereochemistry_preserved_accepts_matching_conformer():
    Chem = pytest.importorskip('rdkit.Chem')
    from rdkit.Chem import AllChem
    from clipdock.utils.mol import stereochemistry_preserved

    mol = Chem.AddHs(Chem.MolFromSmiles('C[C@H](O)CC'))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    assert stereochemistry_preserved(mol) is True


def test_stereochemistry_preserved_rejects_mirrored_conformer():
    Chem = pytest.importorskip('rdkit.Chem')
    from rdkit.Chem import AllChem
    from clipdock.utils.mol import stereochemistry_preserved

    mol = Chem.AddHs(Chem.MolFromSmiles('C[C@H](O)CC'))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (pos.x, pos.y, -pos.z))
    assert stereochemistry_preserved(mol) is False


def test_mol_gen_3d_retries_until_stereochemistry_is_preserved(monkeypatch):
    Chem = pytest.importorskip('rdkit.Chem')
    from clipdock.utils import mol as mol_utils

    checks = {'n': 0}

    def fake_check(mol):
        checks['n'] += 1
        return checks['n'] >= 5

    monkeypatch.setattr(mol_utils, 'stereochemistry_preserved', fake_check)
    out = mol_utils.mol_gen_3d(Chem.MolFromSmiles('CCO'))
    assert out is not None
    assert out.GetNumConformers() == 1
    assert checks['n'] == 5


def test_mol_gen_3d_returns_none_when_stereochemistry_stays_flipped(monkeypatch):
    Chem = pytest.importorskip('rdkit.Chem')
    from clipdock.utils import mol as mol_utils

    monkeypatch.setattr(mol_utils, 'stereochemistry_preserved', lambda mol: False)
    assert mol_utils.mol_gen_3d(Chem.MolFromSmiles('CCO')) is None
