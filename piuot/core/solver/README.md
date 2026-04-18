## Solver

This folder contains the low-level SDE solver stack used internally by PIUOT.

Files:
- `sde.py`
  - forward and adjoint SDE integration entry
- `adjoint_sde.py`
  - adjoint-based backward differentiation
- `methods/`
  - Euler stepping, Brownian noise objects, tensor utility helpers, and contract checks

Practical rule:
- if you only want to run trajectory reconstruction, you can ignore this folder
- only edit this layer if you are modifying the numerical solver itself

