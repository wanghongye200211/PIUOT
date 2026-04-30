## PIUOT Reconstruction Guide

This folder is the main trajectory-reconstruction part of the repository.

If your `.h5ad` does not already contain a latent representation, run `embedding/run_embedding.py` first.

## If You Only Want To Run The Main Model

Run this:

```bash
python piuot/train.py --config piuot/configs/default.yaml
```

For most users, this is the only command that matters.

## File Guide

- `train.py`
  - main entry for training and trajectory reconstruction
  - reads the YAML config, loads the latent representation from your `.h5ad`, trains the model, runs evaluation, and optionally writes plots
- `evaluate.py`
  - computes rollout fit metrics such as transport-style distances
  - use this when you want to re-evaluate a finished run
- `export_trajectory_points.py`
  - exports observed points, predicted rollout points, and dense trajectory lines to CSV
  - this is the stable data interface used by downstream trajectory figures
- `plot.py`
  - draws reconstruction trajectory figures in `PCA / tSNE / UMAP`
  - this is the main plotting script behind the reconstruction panel
- `diagnose.py`
  - checks mass consistency across time
  - use this when you want to see whether predicted relative mass tracks observed mass
- `yaml_config.py`
  - parses the YAML config and resolves device / embedding settings
- `configs/`
  - configuration files
- `input/`
  - where your input `.h5ad` should go
- `core/`
  - internal model implementation
  - most users do not need to edit this folder

## Minimal Workflow

1. Put your `.h5ad` into `piuot/input/`.
2. Edit `piuot/configs/default.yaml`.
3. If needed, build the latent representation first:
   - `python embedding/run_embedding.py --config piuot/configs/default.yaml`
4. Run `python piuot/train.py --config piuot/configs/default.yaml`.
5. Look in `piuot/output/` for the generated run directory and exported trajectory CSV.
6. If needed, re-run:
   - `export_trajectory_points.py` for downstream-ready observed/predicted/trajectory CSV files
   - `plot.py` for trajectory figures
   - `diagnose.py` for mass diagnostics
   - `evaluate.py` for metrics

## Output Location

Outputs are written under:

```text
piuot/output/{run_name}/...
```

The exact subfolder name also includes the training hyperparameter tag.

## What You Usually Need To Edit

Most users only need to edit:
- `experiment.run_name`
- `device.type`
- `data.path`
- `reduction.method` + `reduction.epoch`
- `embedding.input_key`
- `embedding.train_epochs`
- `data.time_key`
- `data.raw_time_key`
- `training.train_epochs`

If you are not changing the model itself, you can ignore most files in `core/`.
