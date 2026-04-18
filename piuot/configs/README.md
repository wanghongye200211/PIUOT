## Configs

This folder stores YAML configuration files for trajectory reconstruction.

## Main File

- `default.yaml`
  - the main config file used by `python piuot/train.py --config piuot/configs/default.yaml`

## What Each Section Means

- `experiment`
  - `name`: human-readable dataset name
  - `run_name`: folder name used under `piuot/output/`
- `device`
  - `type`: one of `cpu`, `cuda`, or `mps`
- `seed`
  - random seed
- `data`
  - `path`: path to your `.h5ad`
  - `embedding_key`: latent representation key in `adata.obsm`; if left empty, the code uses `reduction.method + reduction.epoch`
  - `time_key`: grouping column used to split cells into snapshots
  - `raw_time_key`: actual numeric time used for plotting/reporting
  - `state_key`, `fate_key`: optional columns for downstream analyses
  - `label`: short dataset label
- `reduction`
  - `method`: `gae` or `gaga`
  - `epoch`: used to construct default embedding keys such as `X_gae15` or `X_gaga10`
- `selection`
  - `checkpoint_epoch`: which checkpoint to use later; `auto` is usually fine
- `training`
  - all model and optimization settings
  - most users can keep these defaults unless they are intentionally tuning the model

## Practical Rule

If you are just trying to run the model on a new dataset, edit only:
- `data.path`
- `data.embedding_key` or `reduction.*`
- `data.time_key`
- `data.raw_time_key`
- `device.type`
- `experiment.run_name`
