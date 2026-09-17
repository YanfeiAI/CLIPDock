import os
from pathlib import Path

from clipdock import __version__

VERSION = __version__

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INSTALL_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = INSTALL_ROOT

DOCKING_PARAMS_PATH = str(PACKAGE_ROOT / 'escore.xml')
GAUSS_PARAMS_PATH = str(PACKAGE_ROOT / 'gscore.csv')
VS_MODEL_WEIGHTS_PATH = str(INSTALL_ROOT / 'model' / 'vs_model.ckpt')

_DEFAULT_DATA_ROOT = INSTALL_ROOT / 'data'
if not (INSTALL_ROOT / 'pyproject.toml').is_file():
    _DEFAULT_DATA_ROOT = Path.cwd() / 'data'
DATA_ROOT = Path(os.environ.get('CLIPDOCK_DATA_ROOT', _DEFAULT_DATA_ROOT)).expanduser().resolve()
DATA_RAW_DIR = str(DATA_ROOT / 'raw')
DATA_DOCKED_DIR = str(DATA_ROOT / 'docked')
DOCK_SCORE_K = 0.25

DIST_CUTOFF = 8.0
PAD_SIZE = 8.0

METALS = {'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
          'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce',
          'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
          'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'}
