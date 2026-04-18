## _brownian

This folder is about **Brownian motion**, not biological branches.

In this repository, "brownian" means the random noise process used inside the stochastic differential equation solver.

## What It Does

- generates Brownian increments for SDE simulation
- caches interval queries so repeated calls are consistent and efficient
- supports derived objects such as Brownian paths and trees
- provides the stochastic noise term required by the solver

## File Guide

- `brownian_base.py`
  - abstract base interface for Brownian samplers
- `brownian_interval.py`
  - interval-based Brownian sampler with caching and splitting logic
- `derived.py`
  - helper classes such as `BrownianPath`, `BrownianTree`, and reverse Brownian wrappers
- `__init__.py`
  - exports the public Brownian classes

## Do End Users Need To Understand This?

Usually no.

If you only want to reconstruct trajectories on your own dataset, you can ignore this folder completely.

## Why It Exists

PIUOT solves a stochastic differential equation.  
That requires a noise source.  
This folder is the internal machinery that produces that noise in a mathematically consistent way.
