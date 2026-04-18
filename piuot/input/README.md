## Input Data

Put the user-provided input file here, for example:

- `input.h5ad`

## Required Format

The main input format is `AnnData` (`.h5ad`).

Required content:
- one latent representation
  - usually in `adata.obsm`
  - examples: `X_gae15`, `X_gaga15`
- one grouping time column in `adata.obs`
  - this is `time_key`
  - examples: `time_bin`, `day_index`
- one raw numeric time column in `adata.obs`
  - this is `raw_time_key`
  - examples: `t`, `day`

Special case:
- if `data.embedding_key` is set to `X`, the code uses `adata.X` directly instead of `adata.obsm[...]`

## Time Columns Explained

- `time_key`
  - used to split cells into snapshots
  - the code groups cells by unique values of this column
  - cells with the same `time_key` are treated as one time point
- `raw_time_key`
  - used as the actual numeric time attached to each snapshot
  - mainly affects plotting and reporting
  - if this column is missing, the code falls back to `time_key`

## Practical Time Rule

Your cells do **not** need to be physically sorted in the file.

The code will:
1. read all cells
2. group them by `time_key`
3. sort the unique time groups
4. assign one snapshot per unique time value

That means:
- every cell must belong to exactly one valid time group
- `time_key` should be numeric or at least convertible to numeric
- you need at least two unique time points
- if your real times are already numeric days such as `0, 1, 1.5, 2, 3`, you can use the same values in both columns

## Minimal Example

Your `.h5ad` should look like:

- `adata.obsm["X_gae15"]`: shape `(n_cells, latent_dim)`
- `adata.obs["time_bin"]`: values like `0, 1, 2, 3`
- `adata.obs["t"]`: values like `0.0, 1.0, 1.5, 2.0`

## Optional Columns

These are only needed for downstream analysis, not for basic trajectory reconstruction:
- `obs[state_key]`
- `obs[fate_key]`
