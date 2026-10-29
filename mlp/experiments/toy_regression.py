import numpy as np
from ..regression import MLPRegression
from ..trainer import Trainer
import sys
import time


def test_regression():
    weight_decay = int(sys.argv[1])
    depth = int(sys.argv[2])
    hidden_layer_size = int(sys.argv[3])

    features = np.array(([3, 4], [5, 1], [10, 2]))
    y = np.array(([75], [82], [93]), dtype=float)
    y /= 100

    neural_network = MLPRegression(features, hidden_layer_size, depth, y, weight_decay)

    neural_network.getArchitecture()

    print(features)

    print("Weights before training:\n", neural_network.weights)

    trainer = Trainer(neural_network, features, y)

    t0 = time.perf_counter()

    trainer.train(features, y)

    t1 = time.perf_counter()

    print("Cost function optimisation:\n", trainer.J)

    print("Weights after training:\n", neural_network.weights)

    neural_network.feedForward(neural_network.weights, neural_network.sum_of_weighted_inputs,
                               neural_network.activated_outputs)

    print("Expected output:\n", y)

    print("Output after training\n", neural_network.activated_outputs[-1])

    print("Time taken = ", t1-t0)

if __name__ == test_regression():
    test_regression()

