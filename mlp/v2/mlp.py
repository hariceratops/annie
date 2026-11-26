from typing import Iterable, Iterator, Callable
import numpy as np
from dataclasses import dataclass, field, fields
from itertools import pairwise, starmap
from numpy.typing import NDArray
from scipy import optimize
from mlp.v2.activations import RELU, Activated, PreActivation


@dataclass(frozen=True)
class Slice:
    start: int
    end: int

    def __repr__(self) -> str:
        return f"[{self.start}:{self.end}]"

    def __iter__(self):
        return (getattr(self, f.name) for f in fields(self))

    # merge interval and sub slice for ex bias + weight


# a better name?
def slice(iterable: Iterable, offset: int = 0) -> Iterator[Slice]:
    iterator = iter(iterable)
    for b in iterator:
        end = offset + b
        yield Slice(offset, end)
        offset = end


# a better name?
def slice_w(iterable: Iterable, offset: int = 0) -> Iterator[Slice]:
    iterator = pairwise(iterable)
    for b in iterator:
        end = offset + (b[0] * b[1])
        yield Slice(offset, end)
        offset = end


@dataclass(frozen=True)
class MLPTopology:
    size_in: int
    size_hidden: tuple[int]
    size_out: int
    layers: tuple[int] = field(init=False)
    net_size: int = field(init=False)

    def __post_init__(self):
        layers = [self.size_in, *self.size_hidden, self.size_out]
        neuron_count = self.size_in + sum(self.size_hidden) + self.size_out
        weights_count = sum(starmap(lambda x, y: x * y, pairwise(layers)))
        biases_count = sum(self.size_hidden) + self.size_out
        # z_counts = a_counts = del_counts = neuron_count
        # Twice weights and biases to account gradients
        net_size = (3 * neuron_count) + (2 * (weights_count + biases_count))

        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "net_size", net_size)


# a better name?
@dataclass(frozen=True)
class L:
    slice: Slice
    slices: tuple[Slice, ...]

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Slice:
        return self.slices[idx]

    @classmethod
    def make(cls, layers: Iterable[int], fn: Callable[[Iterable[int], int], Iterator[Slice]], starting_at) -> "L":
        slices = tuple(fn(layers, starting_at))
        slice = Slice(slices[0].start, slices[-1].end)
        return cls(slice, slices)
    
    @classmethod
    def make_inputs(cls, topology: MLPTopology, starting_at) -> "L":
        return cls.make(topology.layers, slice, starting_at)
    
    @classmethod
    def make_biases(cls, topology: MLPTopology, starting_at) -> "L":
        return cls.make(topology.layers[1:], slice, starting_at)
 
    @classmethod
    def make_weights(cls, topology: MLPTopology, starting_at) -> "L":
        return cls.make(topology.layers, slice_w, starting_at)


class NeuralNetBuffer:
    def __init__(self, net_size: int) -> None:
        self.net_size = net_size
        self.buffer = np.random.default_rng().random(net_size)

    def __len__(self) -> int:
        return len(self.buffer)
    
    # what is the return type
    def __getitem__(self, slice: Slice):
        return self.buffer[slice.start: slice.end]
    
    def __setitem__(self, slice: Slice, item):
        self.buffer[slice.start: slice.end] = item
       

class MLP:
    def __init__(self, mlp_topology: MLPTopology) -> None:
        self.net = NeuralNetBuffer(mlp_topology.net_size)
        self.weight_shapes = list(pairwise(mlp_topology.layers))
        # todo a function chainer with chaining element as lambda?
        # todo rename input to layer since output is part of it too
        self.z = L.make_inputs(mlp_topology, 0)
        self.a = L.make_inputs(mlp_topology, self.z.slice.end)
        self.biases = L.make_biases(mlp_topology, self.a.slice.end)
        self.weights = L.make_weights(mlp_topology, self.biases.slice.end)
        # todo inputs? check dimension
        self.delta = L.make_inputs(mlp_topology, self.weights.slice.end)
        self.bias_gradients = L.make_biases(mlp_topology, self.delta.slice.end)
        self.weight_gradients = L.make_weights(mlp_topology, self.bias_gradients.slice.end)
        self.param_slice = Slice(self.biases.slice.start, self.weights.slice.end)
        self.gradient_slice = Slice(self.bias_gradients.slice.start, self.weight_gradients.slice.end)

    # todo abstract training data into separate type
    # todo resolve type hint error
    def x(self, nn_parameters, nn_input, nn_ground_truth) -> tuple[float, NDArray]:
        self.net[self.param_slice] = nn_parameters
        error = 0
        loss = 0

        for sample, ground_truth in zip(nn_input, nn_ground_truth):
            self.net[self.a[0]] = sample

            for i in range(1, len(self.z)):
                m, n = self.weight_shapes[i - 1]
                x = self.net[self.a[i - 1]]
                b = self.net[self.biases[i - 1]]
                W = self.net[self.weights[i - 1]].reshape(m, n)

                self.net[self.z[i]] = (x @ W) + b
                self.net[self.a[i]] = RELU.forward(PreActivation(self.net[self.z[i]]))

            y_hat = self.net[self.a[-1]]
            error = y_hat - ground_truth
            self.net[self.delta[-1]] = 2 * error
            loss += 2 * np.sum(error ** 2)

            for i in list(reversed(range(len(self.weights)))):
                m, n = self.weight_shapes[i]
                self.net[self.weight_gradients[i]] = np.outer(self.net[self.a[i]], self.net[self.delta[i + 1]])
                self.net[self.bias_gradients[i]] = self.net[self.delta[i + 1]]
                if i > 1:
                    self.net[self.delta[i]] = RELU.backward(
                        PreActivation(self.net[self.z[i]]), Activated(self.net[self.a[i]]),
                        self.net[self.weights[i]].reshape(m, n) @ self.net[self.delta[i + 1]]
                    )

        gradients = self.net[self.gradient_slice]
        return (error, gradients)


# class MLPTrainer:
#     def __init__(self, gradient_descent_scheme, max_iter) -> None:
#         self.gradient_descent_scheme = gradient_descent_scheme;
#         self.max_iter = max_iter
#
#     def fun(self) -> None:
#         pass
#
#     def callback(self) -> None:


if __name__ == "__main__":
    m = MLP(MLPTopology(4, (6,), 2))

    # todo use correct io with possibly the abstracted mlpdata class
    nn_inputs = np.random.default_rng().random(4)
    nn_ground_truth = np.random.default_rng().random(4)

    nn_parameters = m.net[m.param_slice]
    options = {'maxiter': 2000, 'disp': True}
    def wrapper_x(nn_parameters, nn_input, nn_ground_truth) -> tuple[float, NDArray]:
        return m.x(nn_parameters, nn_input, nn_ground_truth)

    res = optimize.minimize(
            wrapper_x, 
            nn_parameters, 
            jac=True, 
            method='BFGS',
            args=(nn_inputs, nn_ground_truth), 
            options=options, 
            callback=None # use for debugging is needed
          )

