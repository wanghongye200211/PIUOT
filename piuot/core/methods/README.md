## Methods

This folder contains low-level numerical utilities used by the SDE solver.

These files are not dataset-specific.

## File Guide

- `Euler.py`
  - Euler-style integration utilities for the SDE solver
- `baseFunc.py`
  - shared numerical helper functions
- `checkFunc.py`
  - runtime checks and validation helpers
- `misc.py`
  - small supporting utilities
- `types.py`
  - shared type aliases
- `_brownian/`
  - Brownian-motion utilities used to generate stochastic increments during SDE simulation

## Should You Edit This Folder?

Usually no.

Only touch this folder if you are changing the numerical solver itself.
