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
- `sde.py`
  - stochastic differential equation solver wrapper used by the model
- `adjoint_sde.py`
  - adjoint-based differentiation utilities for SDE training
- `mio_losses.py`
  - transport / fitting losses used during training and evaluation
- `emd.py`
  - Earth Mover's Distance helper code
- `config_Veres.py`
  - legacy dataset-specific config example
- `config_Weinreb.py`
  - legacy dataset-specific config example
- `methods/`
  - low-level SDE solver helper functions

## What To Ignore

If your goal is only to train on your own latent data, you can treat this folder as implementation detail and leave it untouched.
