# Project: annie

## Context
A personal learning repository for neural networks, the first step toward
physical AI. It holds mathematical foundations, from-scratch
implementations, and experiments, organised as independent learning tracks
at the repo root. Host-side Python only; nothing here targets hardware.

The code in the learning tracks is written by hand, by the owner, as the
learning exercise itself. Agents explain, review, and point out bugs in
it, but do not write or complete implementation code there unless
explicitly asked; repo plumbing (tooling, config, docs) is fair game.

Python 3.12, managed with uv (`pyproject.toml`, `uv.lock`); one shared
environment for all tracks. Run everything through `uv run`.
Tracks are top-level directories, not an installed package: `micrograd/`
(Jupyter notebook following Karpathy's micrograd, checked against PyTorch)
and `mlp/` (NumPy MLP with hand-derived backprop, trained with SciPy BFGS).
Scripts in `mlp/experiments/` run as modules from the repo root, e.g.
`uv run python -m mlp.experiments.toy_regression 0 1 3`.
The micrograd notebook needs the Graphviz `dot` binary on PATH.
There is no test suite and no formatter or linter configuration yet.

## Overrides
- Comments: code and notebooks may carry comments that explain *what* the
  code computes, not only *why*. This replaces the default rule that comments
  explain only non-obvious rationale and never restate what the code does.
  The repo exists to learn the maths behind neural networks, so tying a line
  to the equation, derivation step, or array shape it implements is the
  point. This does not license narrating trivial lines (`# increment
  counter`).
