## Embedding

This folder is the dimensionality-reduction step that comes before PIUOT trajectory reconstruction.

It puts the two public embedding builders in one place:
- `GAE`
- `GAGA`

The method is selected from the same YAML used by `piuot/`.

## Main Entry

If you want the repository to run as a complete chain, start here:

```bash
python embedding/run_embedding.py --config piuot/configs/default.yaml
```

This command:
- reads your `.h5ad`
- takes the input matrix from `adata.X`, `adata.obsm[...]`, or `adata.layers[...]`
- trains either `GAE` or `GAGA`
- writes a latent key such as `X_gae15` or `X_gaga15` back into your `.h5ad`

After that, run:

```bash
python piuot/train.py --config piuot/configs/default.yaml
```

## File Guide

- `run_embedding.py`
  - the unified entry
  - reads `reduction.method` from YAML and dispatches to `GAE` or `GAGA`
- `train_gae.py`
  - a generic autoencoder embedding builder
  - optimizes reconstruction loss
- `train_gaga.py`
  - a geometry-aware autoencoder embedding builder
  - optimizes reconstruction loss plus a pairwise-distance preservation term
- `models.py`
  - shared autoencoder backbone
- `common.py`
  - shared YAML / AnnData / device / output helpers

## YAML Fields Used Here

These settings are still stored under `piuot/configs/default.yaml`.

- `device.type`
  - `cpu`, `cuda`, or `mps`
- `data.path`
  - input `.h5ad`
- `data.embedding_key`
  - optional explicit output key to write
  - if empty, the code falls back to `X_gae{reduction.epoch}` or `X_gaga{reduction.epoch}`
- `reduction.method`
  - `gae` or `gaga`
- `reduction.epoch`
  - the historical version tag used in keys like `X_gae15`
  - if `embedding.latent_dim` is not set, it is also used as the default latent dimension
- `embedding.*`
  - controls the actual embedding training

## Input Requirement

For the public GitHub version, the embedding step does not assume any specific dataset.

You only need:
- one `.h5ad`
- one usable feature matrix

By default the feature matrix is `adata.X`.

You can also use:
- `embedding.input_key: some_obsm_key`
- `embedding.input_key: some_layer_key`

The script searches in this order:
1. `adata.X` when `input_key == X`
2. `adata.obsm[input_key]`
3. `adata.layers[input_key]`

## Output Rule

If `embedding.output_path` is not set, the script writes back into the same input `.h5ad`.

That means the simplest workflow is:
1. run `python embedding/run_embedding.py --config piuot/configs/default.yaml`
2. run `python piuot/train.py --config piuot/configs/default.yaml`

If you want to keep the original file untouched, set `embedding.output_path` directly in YAML.

## Practical Rule

- use `GAE` when you just want a simple latent representation
- use `GAGA` when you want the embedding to preserve geometry more explicitly
- if your `.h5ad` already contains a latent representation, you can skip this folder and go straight to `piuot/train.py`
