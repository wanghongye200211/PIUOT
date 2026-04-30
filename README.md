## scPIUOT

This repository is a GitHub-ready scPIUOT code bundle for arbitrary single-cell datasets.

Structure:
- `embedding/`: dimensionality reduction before trajectory reconstruction
  - `run_embedding.py`: unified embedding entry
  - `train_gae.py`: generic autoencoder embedding
  - `train_gaga.py`: geometry-aware autoencoder embedding
- `piuot/`: trajectory reconstruction
  - `train.py`: main training/reconstruction entry
  - `evaluate.py`: trajectory fit metrics
  - `plot.py`: trajectory projection plots
  - `diagnose.py`: mass diagnostics
  - `input/`: user `.h5ad` input location
  - `core/`: internal PIUOT model implementation
- `criticality/`: metric export helpers used by downstream plots
- `downstream/`: downstream fate, perturbation, and criticality plotting
- `piuot/configs/default.yaml`: shared training config

What is intentionally excluded:
- trained checkpoints
- `.h5ad` data
- ad hoc output folders
- large intermediate arrays

Generated figure/data bundles are not committed. Downstream figures are rebuilt
from CSV files exported by the reconstruction and downstream analysis scripts.

Usage:
1. Put your `.h5ad` input under `piuot/input/`.
2. Edit `piuot/configs/default.yaml`, or start from one selected preset:
   - `piuot/configs/ipsc_day0to5_official_gae15.yaml`
   - `piuot/configs/gse75748_oldprofile_gaga5.yaml`
3. Set `device.type` to `mps`, `cuda`, or `cpu`.
4. Choose your embedding setting in `reduction.method` and `reduction.epoch`.
5. If your `.h5ad` does not already contain a latent representation, build it first:
   - `python embedding/run_embedding.py --config piuot/configs/default.yaml`
   - by default this writes the latent key back into the same `.h5ad`
6. If your `.h5ad` already uses a custom embedding name, set `data.embedding_key` directly and skip the embedding step.
7. Set `data.time_key` and `data.raw_time_key`.
8. Train and reconstruct trajectories:
   - `python piuot/train.py --config piuot/configs/default.yaml`

Start here if you are confused about the structure:
- `embedding/README.md`
- `piuot/README.md`
- `piuot/input/README.md`
- `piuot/configs/README.md`

Criticality and downstream:
- no YAML is used there
- open the corresponding `.py` script and manually edit the run name, data path, label, checkpoint, and device defaults near the top
- then run the script directly, for example:
  - `python criticality/compute_original_qreshape_mass_indicator.py`
  - `python criticality/compare_potential_related_indicators.py`
  - `python downstream/run_downstream.py`

Suggested figure sequence for GitHub or paper assembly:
- `Figure b`: continuous dynamics
  - observed cells, predicted cells, and dense rollout trajectories in one consistent view
- `Figure c`: potential landscape state map
  - a selected coordinate-space `3D` terrain view
- `Figure D`: multi-model comparison
  - quantitative panels such as `W1`, `W2^2`, `MMD`, plus a shared manifold overlay
- `Figure e`: warning UMAP landscape
  - map the warning score back to the UMAP manifold with low values shown in light colors

Public figure builders:
- `Figure b`: `piuot/export_trajectory_points.py` then `downstream/figure_b_continuous_dynamics.py`
- `Figure c`: `downstream/export_potential_landscape_points.py` then `downstream/figure_c_umap_potential_landscape.py`
- `Figure D benchmark metrics`: `downstream/build_figure_d_benchmark_metrics.py`
- `Figure e`: `downstream/figure_e_warning_umap.py`
- `Figure G`: `downstream/figure_g_gene_perturbation_screen.py`

Practical rule:
- use `embedding/` to produce a latent representation on arbitrary new datasets
- keep `piuot/` generic and YAML-driven
- treat `criticality/` and `downstream/` as manual analysis templates that you adapt to your current run
- use the downstream scripts to rebuild your own figure sequence for your dataset instead of relying on pre-bundled outputs

## License

This project is released under the MIT License. See `LICENSE` for details.
