# annie

Learning neural networks from first principles, as the first step toward
physical AI: mathematical foundations, from-scratch implementations, and
experiments.

## Tracks

| Track | What it is |
|---|---|
| [`micrograd/`](micrograd/) | A scalar autograd engine and tiny MLP built up step by step in a notebook, following Karpathy's micrograd. Checked against PyTorch. |
| [`mlp/`](mlp/) | A from-scratch NumPy MLP with hand-derived backprop, trained with SciPy's BFGS. Experiments live in `mlp/experiments/`. |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Graphviz (`dot`), which the
micrograd notebook uses to render computation graphs.

```sh
uv sync
```

## Running

The micrograd notebook: open `micrograd/micrograd.ipynb` and select the
`.venv` kernel, or

```sh
uv run --with jupyterlab jupyter lab
```

MLP experiments take `weight_decay depth hidden_layer_size` as arguments:

```sh
uv run toy-regression 0 1 3
uv run iris 0 1 3
```

A NumPy layout benchmark (flat parameter buffer viewed as a matrix vs
slicing rows) takes no arguments; its results and reasoning are recorded in
its docstring:

```sh
uv run matvec-layout
```
