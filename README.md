# CLIPDock

*CLIPDock: a contrastive learning-based interatomic potential model
for accurate protein–ligand docking*

CLIPDock combines an empirical interatomic potential with a
contrastive learning-based interatomic potential model for molecular
docking, and provides a graph-based model for rescoring in virtual
screening.

An online docking service is available at [clipdock.cn](https://clipdock.cn).

## Installation

CLIPDock requires Python 3.11.

```bash
git clone https://github.com/YanfeiAI/CLIPDock.git
cd CLIPDock
conda env create -f environment.yml
conda activate clipdock
python -m pip install -e ".[all]"
clipdock --version
```

For docking only, without the PyTorch dependencies, use
`python -m pip install -e .` instead.

## Usage

A reference ligand is recommended to define the binding pocket.

Dock an SDF or MOL2 ligand:

```bash
clipdock \
  -r data/9iwy_protein.pdb \
  -l data/9iwy_ligand.sdf \
  --ref data/9iwy_ligand.sdf \
  -o data/9iwy_docked_ligand.sdf
```

Dock a ligand from a SMILES string:

```bash
clipdock \
  -r data/9iwy_protein.pdb \
  -s "c1c(S(=O)(=O)C(C)C)ccc2nccc(Nc3cc4nc(sc4cc3)C3CCCC3)c12" \
  --ref data/9iwy_ligand.sdf \
  -o data/9iwy_docked_ligand.sdf
```

Prepare a screening example library:

```bash
python script/prepare_screening_examples.py
```

The generated `data/raw/screening_examples.csv` contains diverse small-molecule
compounds with names and SMILES.

```bash
clipdock vs \
  -i data/raw/screening_examples.csv \
  -o data/docked/9iwy_vs/01_clipdock \
  -r data/9iwy_protein.pdb \
  --ref data/9iwy_ligand.sdf
```

Rescore all docked ligands with VS-score:

```bash
clipdock vs-score \
  -p data/docked/9iwy_vs/01_clipdock/9iwy_protein_pocket.pdb \
  --lig_dir data/docked/9iwy_vs/01_clipdock/pose_r32 \
  -o data/docked/9iwy_vs/02_vs_score/vs_scores \
  --top_num 100 \
  --top_dir data/docked/9iwy_vs/02_vs_score/top100 \
  --device 0
```

This writes all VS-score results to `02_vs_score/vs_scores.csv` and the selected
top 100 poses to `02_vs_score/top100/`.

Lower CLIPDock scores and higher VS-scores are better. Use `--device -1` for
CPU VS-score inference.

Before using VS-score, download `vs_model.ckpt` from
[GitHub Releases](https://github.com/YanfeiAI/CLIPDock/releases) and place it in
the `model` directory. Use `--model` to load it from another location.

## Reproduction datasets

- [`PDBPL`](https://doi.org/10.5281/zenodo.22766987): 41,275 training and 783
  validation complexes
- [`CLIPDock test datasets`](https://doi.org/10.5281/zenodo.22767165): fragment
  (331), nucleic-acid-associated (146), and FABP4 (179) complexes

Download the two releases and prepare them from the repository root:

```bash
python script/prepare_data.py \
  --pdbpl-dir /path/to/PDBPL \
  --test-data-dir /path/to/CLIPDock_test_datasets
```

## Reproducing the paper

### Training

Train G-score using the paper configuration:

```bash
python -m model.train \
  --train_split pdbpl_train \
  --val_split pdbpl_val \
  --data_suffix r8p32 \
  --devices 0 \
  --restarts 8 \
  --pose_num 32 \
  --epoch 5
```

This exports `gscore.csv` to `data/training/gscore`. Pass
`data/training/gscore/gscore.csv` to `clipdock` or `clipdock vs` with
`--gscore`.

The generated docking poses are then used to train VS-score:

```bash
python -m model.train_vs \
  --batch_size 64 \
  --lr 1e-3 \
  --epochs 600 \
  --device 0 \
  --data_suffix r8p32
```

The final checkpoint is `data/training/vs_score/last.ckpt`; pass it with
`--model` to `clipdock vs-score`.

Benchmark commands use the bundled models by default. To use newly trained
models, pass `--gscore data/training/gscore/gscore.csv`; for virtual-screening
enrichment, also pass `--model data/training/vs_score/last.ckpt`.

### Docking benchmarks

Evaluate the three CLIPDock test datasets with:

```bash
for split in fragment_test nucleic_acid_associated_test fabp4_test; do
  python -m script.test_dock \
    --split "$split" --suffix _clipdock \
    --restarts 32 --pose_num 32 --repeat 3 --bust \
    --metrics_output "${split}_clipdock_metrics.csv"
done
```

`--metrics_output` writes a clean summary CSV.

Prepare the public PoseBusters V2 and PoseX benchmarks from their official
sources, then evaluate them using the same protocol:

```bash
python script/prepare_benchmarks.py posebusters posex_sd posex_cd

for split in posebusters posex_sd posex_cd; do
  python -m script.test_dock \
    --split "$split" --suffix _clipdock \
    --restarts 32 --pose_num 32 --repeat 3 --bust \
    --metrics_output "${split}_clipdock_metrics.csv"
done
```

The evaluator reports RMSD-based Top-1, Top-5 and best-pose success rates, together
with the combined RMSD < 2 Å and PB-valid success rate. The PoseBusters
evaluation uses the official 308-complex subset from
[Zenodo record 8278563](https://zenodo.org/records/8278563); PoseX is obtained
from the official [PoseX dataset](https://huggingface.co/datasets/CataAI/PoseX).

### Affinity ranking

The affinity benchmark uses 16 congeneric ligand series from the
[public binding free energy benchmark](https://github.com/schrodinger/public_binding_free_energy_benchmark/tree/v1.0):

```bash
python script/prepare_affinity.py
python script/affinity.py
```

### Virtual-screening enrichment

The virtual-screening benchmark uses the public TrueDecoy and RandomDecoy
datasets from
[Zenodo record 14874127 (v3)](https://zenodo.org/records/14874127):

```bash
python script/prepare_vs.py
python script/vs.py all --device 0
```

## License

The CLIPDock source code and released model weights are distributed under the
MIT License. External benchmark datasets retain their upstream licenses.
