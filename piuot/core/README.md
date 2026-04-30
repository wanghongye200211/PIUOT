## Core

This folder contains the internal PIUOT model implementation.

Most users do not need to edit anything here.

## File Guide

- `config_model.py`
  - internal argument/config builder
  - loads `.h5ad` / `.pt` data and splits cells by time
- `train.py`
  - main training loop and checkpoint writing
- `evaluation.py`
  - rollout evaluation utilities
- `model.py`
  - neural trajectory model and drift / growth definitions
- `solver/`
  - low-level SDE solver stack
  - contains the actual forward/adjoint integration code and Brownian helpers
- `mio_losses.py`
  - transport / fitting losses used during training and evaluation
- `emd.py`
  - Earth Mover's Distance helper code

## What To Ignore

If your goal is only to train on your own latent data, you can treat this folder as implementation detail and leave it untouched.
