from typing import Iterable, Iterator, Callable
import numpy as np
from dataclasses import dataclass, field, fields
from itertools import pairwise, starmap


@dataclass(frozen=True)
class Slice:
    start: int
    end: int

    def __repr__(self) -> str:
        return f"[{self.start}:{self.end}]"

    def __iter__(self):
        return (getattr(self, f.name) for f in fields(self))


def slice(iterable: Iterable, offset: int = 0) -> Iterator[Slice]:
    iterator = iter(iterable)
    for b in iterator:
        end = offset + b
        yield Slice(offset, end)
        offset = end


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
        inputs_count = self.size_in + sum(self.size_hidden) + self.size_out
        weights_count = sum(starmap(lambda x, y: x * y, pairwise(layers)))
        biases_count = sum(self.size_hidden) + self.size_out
        net_size = inputs_count + (2 * (weights_count + biases_count))

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
        # a function chainer with chaining element as lambda?
        # rename input to layer since output is part of it too
        self.inputs = L.make_inputs(mlp_topology, 0)
        self.biases = L.make_biases(mlp_topology, self.inputs.slice.end)
        self.weights = L.make_weights(mlp_topology, self.biases.slice.end)
        self.bias_gradients = L.make_biases(mlp_topology, self.weights.slice.end)
        self.weight_gradients = L.make_weights(mlp_topology, self.bias_gradients.slice.end)
        
        # create slices

    def feed_forward(self, nn_input):
        self.net[self.inputs[0]] = nn_input

        for i in range(1, len(self.inputs)):
            m, n = self.weight_shapes[i - 1]
            x = self.net[self.inputs[i - 1]]
            b = self.net[self.biases[i - 1]]
            W = self.net[self.weights[i - 1]].reshape(m, n)

            self.net[self.inputs[i]] = (x @ W) + b

        res = self.net[self.inputs[-1]]
        return res

    def backpropagate(self, y_ground_truth, y_hat):
        # e = sum(2 * sum((y_ground_truth - y_hat) ** 2))

        for i in list(reversed(range(len(self.inputs)))):
            i_start, i_end = self.inputs[i]
        pass


m = MLP(MLPTopology(4, (6,), 2))
m.feed_forward(np.random.default_rng().random(4))

