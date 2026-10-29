from .network import NeuralNetwork
from . import activations
import numpy as np
import time


class MLPClassifier(NeuralNetwork):
    def __init__(self, data_set, hidden_layer_size, depth, output_data, weight_decay):
        super().__init__(data_set, hidden_layer_size, depth, output_data, weight_decay)

    def defineArchitecture(self):
        super().defineArchitecture()

    def calcNumericalGradients(self, y):
        super().calcNumericalGradients(y)

    def getWeights(self):
        params = super().getWeights()
        return params

    def setWeights(self, params):
        super().setWeights(params)

    def getArchitecture(self):
        super().getArchitecture()

    def feedForward(self, weights, sum_of_weighted_inputs, activated_outputs):
        # t0 = time.perf_counter()
        activated_outputs[0] = self.input_data
        for layer in list(range(1, self.number_of_layers)):
            sum_of_weighted_inputs[layer] = np.dot(activated_outputs[layer - 1], weights[layer - 1])
            activated_outputs[layer] = activations.soft_plus(sum_of_weighted_inputs[layer])
        self.yHat = activated_outputs[-1]
        # t1 = time.perf_counter()
        # print("Time taken for feed forward = ", t1-t0)

    def costFunction(self, y):
        self.feedForward(self.weights, self.sum_of_weighted_inputs, self.activated_outputs)
        j = sum(0.5 * sum((y - self.yHat) ** 2))
        return j

    def calcGradients(self, y):
        # t0 = time.perf_counter()
        self.delta[self.number_of_layers - 1] = (self.yHat - y)

        for layer in range(self.number_of_layers - 2, 0, -1):
            self.delta[layer] = np.dot(self.delta[layer + 1], self.weights[layer].T) * \
                                (self.sigmoid(self.sum_of_weighted_inputs[layer]))

        for layer in list(range(0, self.number_of_layers - 1)):
            self.gradients[layer] = np.dot(self.activated_outputs[layer].T, self.delta[layer + 1])

        # t1 = time.perf_counter()
        # print("Time taken for gradient calc = ", t1-t0)
        # print("Gradients:\n", self.gradients)

        dJdW = np.array([])
        for i in range(len(self.gradients)):
            dJdW = np.append(dJdW, self.gradients[i].flatten())
        return dJdW

