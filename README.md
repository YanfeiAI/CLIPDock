# CLIPDock: a contrastive learning-based interatomic potential model for accurate protein–ligand docking

CLIPDock combines an empirical interatomic potential with a
contrastive learning-based interatomic potential model for molecular
docking, and provides a graph-based model for rescoring in virtual
screening.

An online docking service is available at [clipdock.cn](https://clipdock.cn).

## 1. Installation

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

CLIPDock was tested with Python 3.11.18 on Ubuntu 22.04.5 LTS (x86_64) in two
environments: a server with two AMD EPYC 9654 CPUs and 377 GiB RAM, and a local
computer with an AMD Ryzen 9 7945HX and 32 GB RAM running Ubuntu under WSL2.
Installation on the local computer took approximately 21 min. No GPU is required
for docking. Dependencies are specified in `environment.yml` and `pyproject.toml`.

## 2. Usage

Docking runs on CPU. Remove any co-crystallized ligand from the receptor, and
pass a reference ligand separately with `--ref` to define the binding pocket.

### 2.1 Docking

Dock an SDF or MOL2 ligand:

```bash
clipdock \
  -r data/9iwy_protein.pdb \
  -l data/9iwy_ligand.sdf \
  --ref data/9iwy_ligand.sdf \
  -o data/9iwy_docked_ligand.sdf
```

Expected output: `data/9iwy_docked_ligand.sdf`, containing up to 9 poses with a
`CLIPDock score` property. On the local computer described above, three runs with
the default settings (32 restarts, 9 poses, 8 workers) took 3.4–9.0 s
(median: 3.4 s), including JIT compilation during the first run.
Observed best scores ranged from
-10.257 to -10.119. Results may vary due to stochastic optimization.

Dock a ligand from SMILES:

```bash
clipdock \
  -r data/9iwy_protein.pdb \
  -s "c1c(S(=O)(=O)C(C)C)ccc2nccc(Nc3cc4nc(sc4cc3)C3CCCC3)c12" \
  --ref data/9iwy_ligand.sdf \
  -o data/9iwy_docked_ligand.sdf
```

To use your own data, replace the example receptor (`-r`), ligand (`-l` or `-s`),
and reference ligand (`--ref`) with your own inputs. The reference ligand should
share the receptor's coordinate system.

### 2.2 Virtual screening

Prepare a screening example library:

```bash
python script/prepare_screening_examples.py
```

The generated `data/raw/screening_examples.csv` contains compound names and SMILES.

Run virtual screening:

```bash
clipdock vs \
  -i data/raw/screening_examples.csv \
  -o data/docked/9iwy_vs/01_clipdock \
  -r data/9iwy_protein.pdb \
  --ref data/9iwy_ligand.sdf
```

Results are saved to `results.csv` and `results_sorted.csv` in the output directory.

To screen your own compounds, provide a CSV file with compound names and SMILES
in the first two columns (`name,smiles`), including a header row. Replace `-i`,
`-r`, `--ref`, and `-o` with your input CSV, receptor, reference ligand, and output
directory, respectively.

### 2.3 VS-score rescoring

Before using VS-score, download `vs_model.ckpt` from
[GitHub Releases](https://github.com/YanfeiAI/CLIPDock/releases) and place it in
the `model` directory. Use `--model` to load it from another location.
GPU inference was tested on an NVIDIA GeForce RTX 5090 with driver 580.173.02
and PyTorch 2.8.0 (CUDA 12.8). Use `--device -1` for CPU inference.

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

This writes all VS-score results to `02_vs_score/vs_scores.csv` and up to 100
top-ranked poses to `02_vs_score/top100/`.

To rescore your own docked ligands, provide a receptor pocket PDB with `-p`,
docked ligand SDF files with `--lig_dir` (or a single SDF with `-l`), and an output
prefix with `-o`.

Lower CLIPDock scores and higher VS-scores are better.

## 3. Reproduction datasets

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

## 4. Reproducing the paper

### 4.1 Training

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

The benchmarks below use the default G-score parameters and the released
VS-score checkpoint.
To use newly trained models, pass `--gscore data/training/gscore/gscore.csv`;
for virtual-screening enrichment, also pass
`--model data/training/vs_score/last.ckpt`.

### 4.2 Docking benchmarks

Evaluate the three CLIPDock test datasets with:

```bash
for split in fragment_test nucleic_acid_associated_test fabp4_test; do
  python -m script.test_dock \
    --split "$split" --suffix _clipdock \
    --restarts 32 --pose_num 32 --repeat 3 --bust \
    --metrics_output "${split}_clipdock_metrics.csv"
done
```

`--metrics_output` writes a summary CSV.

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

### 4.3 Affinity ranking

The affinity benchmark uses 16 congeneric ligand series from the
[public binding free energy benchmark](https://github.com/schrodinger/public_binding_free_energy_benchmark/tree/v1.0):

```bash
python script/prepare_affinity.py
python script/affinity.py
```

### 4.4 Virtual-screening enrichment

The virtual-screening benchmark uses the public TrueDecoy and RandomDecoy
datasets from
[Zenodo record 14874127 (v3)](https://zenodo.org/records/14874127):

```bash
python script/prepare_vs.py
python script/vs.py all --device 0
```

## 5. License

The CLIPDock source code and released model weights are distributed under the
MIT License.
