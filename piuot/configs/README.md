## Configs

This folder stores the shared YAML configuration used by both:
- `embedding/`
- `piuot/`

## Main File

- `default.yaml`
  - the main config file used by:
    - `python embedding/run_embedding.py --config piuot/configs/default.yaml`
    - `python piuot/train.py --config piuot/configs/default.yaml`
- `ipsc_day0to5_official_gae15.yaml`
  - selected official iPSC Day0-Day5 reconstruction preset
  - expects `piuot/input/ipsc_direct_gae15_input.h5ad`, or edit `data.path`
  - the referenced h5ad should already contain `X_gae15`, so you can run `python piuot/train.py --config piuot/configs/ipsc_day0to5_official_gae15.yaml` directly
- `gse75748_oldprofile_gaga5.yaml`
  - selected GSE75748 old-profile reconstruction preset
  - expects `piuot/input/gse75748_stage0_pca50_phate_input.h5ad`, or edit `data.path`
  - starts from the processed stage0 h5ad, not the already embedded GAGA h5ad
  - run `python embedding/run_embedding.py --config piuot/configs/gse75748_oldprofile_gaga5.yaml` first to write `X_gaga`
  - then run `python piuot/train.py --config piuot/configs/gse75748_oldprofile_gaga5.yaml`

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
  - for historical compatibility, this value is also used as the default latent dimension when `embedding.latent_dim` is empty
- `embedding`
  - `input_key`: which matrix to embed; `X` means `adata.X`
    - for the GSE75748 stage0 preset this is `X_pca`
  - `output_key`: explicit latent key to write into `adata.obsm[...]`
  - `output_path`: output `.h5ad`; if empty, the embedding is written back into the same input file
  - `latent_dim`: output latent dimension; if empty, falls back to `reduction.epoch`
  - `hidden_dims`: encoder/decoder hidden widths
  - `batch_size`, `train_epochs`, `learning_rate`, `weight_decay`
    - embedding optimization settings
  - `noise_std`
    - optional Gaussian noise used during training
  - `standardize`
    - whether to z-score the input matrix before training
  - `distance_weight`, `reconstruction_weight`
    - mainly affect `GAGA`
- `selection`
  - `checkpoint_epoch`: which checkpoint to use later; `auto` is usually fine
- `training`
  - all model and optimization settings
  - most users can keep these defaults unless they are intentionally tuning the model
  - optional late-schedule fields such as `density_start_epoch`, `action_start_epoch`, and `hjb_start_epoch` are supported by `piuot/train.py`

## Practical Rule

If you are just trying to run the model on a new dataset, edit only:
- `data.path`
- `reduction.*`
- `embedding.input_key`
- `embedding.train_epochs`
- `data.time_key`
- `data.raw_time_key`
- `device.type`
- `experiment.run_name`
