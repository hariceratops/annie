import numpy as np
from scipy import optimize
import time
from .activations import soft_plus, soft_max


class NeuralNetwork(object):
    def __init__(self, data_set, hidden_layer_size, depth, output_data, weight_decay):

        self.input_data = data_set
        self.output_data = output_data

        self.number_of_training_data = data_set.shape[0]

        self.input_layer_size = data_set.shape[1]
        self.output_layer_size = output_data.shape[1]
        self.hidden_layer_size = hidden_layer_size
        self.number_of_layers = depth + 2
        self.depth = depth

        self.weight_decay = weight_decay

        self.total_number_of_nodes = 0
        self.weight_topology = []
        self.node_output_topology = []

        self.activated_outputs = []
        self.sum_of_weighted_inputs = []
        self.weights = []
        self.delta = []
        self.gradients = []

        self.training_iteration = 0

        self.yHat = np.array([])

        self.defineArchitecture()

    def defineArchitecture(self):

        self.total_number_of_nodes = self.input_layer_size + self.hidden_layer_size * self.depth + \
                                     self.output_layer_size

        self.weight_topology.append((self.input_layer_size * self.hidden_layer_size,
                                     (self.input_layer_size, self.hidden_layer_size)))
        for i in list(range(0, self.depth - 1)):
            self.weight_topology.append((self.hidden_layer_size * self.hidden_layer_size,
                                         (self.hidden_layer_size, self.hidden_layer_size)))
        self.weight_topology.append((self.hidden_layer_size * self.output_layer_size,
                                     (self.hidden_layer_size, self.output_layer_size)))

        for i in self.weight_topology:
            self.weights.append(np.random.rand(i[0]))
            self.gradients.append(np.zeros(i[0]))

        count = 0
        for i in self.weights:
            self.weights[count] = np.reshape(self.weights[count], self.weight_topology[count][1])
            self.gradients[count] = np.reshape(self.gradients[count], self.weight_topology[count][1])
            count += 1

        self.weights = np.array(self.weights, dtype=object)
        self.gradients = np.array(self.gradients, dtype=object)

        self.node_output_topology.append((self.number_of_training_data * self.input_layer_size,
                                          (self.number_of_training_data, self.input_layer_size)))
        for i in list(range(0, self.depth)):
            self.node_output_topology.append((self.number_of_training_data * self.hidden_layer_size,
                                              (self.number_of_training_data, self.hidden_layer_size)))

        self.node_output_topology.append((self.output_layer_size * self.number_of_training_data,
                                          (self.number_of_training_data, self.output_layer_size)))

        for i in self.node_output_topology:
            self.sum_of_weighted_inputs.append(np.ones(i[0]))
            self.activated_outputs.append(np.ones(i[0]))
            self.delta.append(np.ones(i[0]))

        self.sum_of_weighted_inputs = np.array(self.sum_of_weighted_inputs, dtype=object)
        self.activated_outputs = np.array(self.activated_outputs, dtype=object)
        self.delta = np.array(self.delta, dtype=object)

        count = 0
        for i in self.sum_of_weighted_inputs:
            self.sum_of_weighted_inputs[count] = np.reshape(self.sum_of_weighted_inputs[count],
                                                            self.node_output_topology[count][1])
            self.activated_outputs[count] = np.reshape(self.activated_outputs[count],
                                                       self.node_output_topology[count][1])
            self.delta[count] = np.reshape(self.delta[count],
                                           self.node_output_topology[count][1])
            count += 1

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def hyberbolic_tangent(self,z):
        return np.tanh(z)

    def hyberbolic_tangent_prime(self, z):
        # Gradient of sigmoid
        return 1 - (z * z)

    def hyberbolic_tangent_prime0(self, z):
        # Gradient of sigmoid
        return 1 - (np.tanh(z) * np.tanh(z))

    def getArchitecture(self):
        print("number_of_training_data = ", self.number_of_training_data)
        print("input_layer_size =  ", self.input_layer_size)
        print("output_layer_size = ", self.output_layer_size)
        print("hidden_layer_size = ", self.hidden_layer_size)
        print("weight_decay = ", self.weight_decay)
        print("number_of_layers = ", self.number_of_layers)
        print("depth = ", self.depth)

        print("total_number_of_nodes = ", self.total_number_of_nodes)
        print("weight_topology = ", self.weight_topology)
        print("node_output_topology = ", self.node_output_topology)

    def feedForward(self, weights, sum_of_weighted_inputs, activated_outputs):
        t0 = time.perf_counter()
        activated_outputs[0] = self.input_data
        count = 0
        for layer in list(range(1, self.number_of_layers)):
            sum_of_weighted_inputs[layer] = np.dot(activated_outputs[layer - 1], weights[layer - 1])
            # activated_outputs[layer] = self.hyberbolic_tangent(sum_of_weighted_inputs[layer])
            activated_outputs[layer] = self.sigmoid(sum_of_weighted_inputs[layer])
            # activated_outputs[layer] = soft_plus(sum_of_weighted_inputs[layer])
            count = layer
        # print(activated_outputs)
        # sum_of_weighted_inputs[-1] = np.dot(activated_outputs[count], weights[count])
        # activated_outputs = soft_max(sum_of_weighted_inputs[-1])
        self.yHat = activated_outputs[-1]
        t1 = time.perf_counter()
        # print("Time taken for feed forward = ", t1-t0)

    def costFunction(self, y):
        self.feedForward(self.weights, self.sum_of_weighted_inputs, self.activated_outputs)
        epsilon = 1e-100
        j = sum(0.5 * sum((y - self.yHat) ** 2))
        # print(j)

        # print("Training iteration = ", self.training_iteration)

        # Todo: Use logarithmic cost function since parabolic cost functions may have convexity problems
        # print("Index in yHat with 0 = ", np.argwhere(self.yHat == 0).ravel())
        first_part_of_cost = np.sum((y) * np.log(self.yHat))

        # print("Index in first part with nan = ", np.argwhere(np.isnan(first_part_of_cost)).ravel())

        # for i in (np.argwhere(np.isnan(first_part_of_cost)).ravel()):
        #     first_part_of_cost[i] = epsilon

        k = 1 - self.yHat
        # g = 1 - y
        # print("Index in 1-y with 0 = ", np.argwhere(g == 0))
        # print("Index in 1-yHat with 0 = ", np.argwhere(k == 0))

        # for i in (np.argwhere(k == 0)):
        #     k[i] = epsilon

        second_part_of_cost = np.sum((1.0 - y) * (np.log(k)))

        # print("Index in second part with nan = ", np.argwhere(np.isnan(second_part_of_cost)))

        # print(first_part_of_cost)
        # print(second_part_of_cost)
        #
        # weights = np.array([])
        # count = 0
        # for i in self.weights:
        #     weights = np.append(weights, self.weights[count].flatten())
        #     count += 1

        # regularization_term = (self.weight_decay / (2.0 * self.number_of_training_data)) * \
        #                       np.sum(np.power(weights, 2))

        # j = ((-(1.0 / self.number_of_training_data)) * np.sum((first_part_of_cost + second_part_of_cost)))
        #                                           + regularization_term

        # print("Cost = ", j, "\n")

        self.training_iteration += 1

        return j

    def calcGradients(self, y):
        t0 = time.perf_counter()
        self.delta[self.number_of_layers - 1] = (self.yHat - y)
        # self.delta[self.number_of_layers - 1] = (self.yHat - y)
        # self.delta[self.number_of_layers - 1] = (self.yHat - y) * (1 - (self.yHat * self.yHat))
        # self.delta[self.number_of_layers - 1] = (self.yHat - y) * (self.yHat * (1 - self.yHat))
                                                  # self.sigmoidPrime(
                                                  # self.sum_of_weighted_inputs[self.number_of_layers - 1])
        # self.delta[self.number_of_layers - 1] = (self.yHat - y) * \
        #                                           self.hyberbolic_tangent_prime0(
        #                                           self.sum_of_weighted_inputs[self.number_of_layers - 1])

        for layer in range(self.number_of_layers - 2, 0, -1):
            # self.delta[layer] = np.dot(self.delta[layer + 1], self.weights[layer].T) * \
            #                     self.sigmoidPrime(self.sum_of_weighted_inputs[layer])

            self.delta[layer] = np.dot(self.delta[layer + 1], self.weights[layer].T) * \
                                (self.activated_outputs[layer] * (1 - self.activated_outputs[layer]))

            # self.delta[layer] = np.dot(self.delta[layer + 1], self.weights[layer].T) * \
            #                     (1 - self.activated_outputs[layer] * self.activated_outputs[layer])

            # self.delta[layer] = np.dot(self.delta[layer + 1], self.weights[layer].T) * \
            #                     (self.sigmoid(self.sum_of_weighted_inputs[layer]))

        for layer in list(range(0, self.number_of_layers - 1)):
            self.gradients[layer] = np.dot(self.activated_outputs[layer].T, self.delta[layer + 1])

        # for i in self.gradients:
        #     i /= self.number_of_training_data

        # self.gradients /= self.number_of_training_data

        # self.gradients *= 2

        t1 = time.perf_counter()
        # print("Time taken for gradient calc = ", t1-t0)

        # print("Gradients:\n", self.gradients)
        # print("Gradients:\n", self.gradients*2)

        dJdW = np.array([])
        for i in range(len(self.gradients)):
            dJdW = np.append(dJdW, self.gradients[i].flatten())

        # print("Flatten\n", dJdW)

        return dJdW

    def getWeights(self):
        params = np.array([])
        for i in range(len(self.weights)):
            params = np.append(params, self.weights[i].flatten())
        return params

    def setWeights(self, params):
        count = 0
        offset = 0
        for number_of_nodes, connections in self.weight_topology:
            k = params[offset: offset + number_of_nodes]
            k = np.reshape(k, connections)
            self.weights[count] = k
            count += 1
            offset = number_of_nodes

    def calcNumericalGradients(self, y):
        epsilon = 1e-4
        weights0 = self.weights
        numerical_gradient = self.weights * 0

        for p in range(len(weights0)):
            for q in range(len((weights0[p]))):
                for r in range(len((weights0[p][q]))):
                    original_weights = weights0[p][q][r]

                    weights0[p][q][r] += epsilon

                    activated_outputs0 = self.activated_outputs * 0
                    sum_of_weighted_inputs0 = self.sum_of_weighted_inputs * 0
                    self.feedForward(weights0, activated_outputs0, sum_of_weighted_inputs0)
                    loss2 = self.costFunction(y)

                    weights0[p][q][r] = original_weights

                    weights0[p][q][r] -= epsilon

                    activated_outputs0 = self.activated_outputs * 0
                    sum_of_weighted_inputs0 = self.sum_of_weighted_inputs * 0
                    self.feedForward(weights0, activated_outputs0, sum_of_weighted_inputs0)
                    loss1 = self.costFunction(y)

                    numerical_gradient[p][q][r] = (loss2 - loss1) / (2 * epsilon)

                    weights0[p][q][r] = original_weights

        print("Numerical Gradients:\n", numerical_gradient)


class Trainer(object):
    def __init__(self, neural_network, X, y):
        # Make Local reference to network:
        self.neural_network = neural_network
        self.J = []
        self.X = X
        self.y = y
        self.optimizationResults = []

    def callbackFunction(self, params):
        self.neural_network.setWeights(params)
        self.J.append(self.neural_network.costFunction(self.y))

    def costFunctionWrapper(self, params, X, y):
        self.neural_network.setWeights(params)
        cost = self.neural_network.costFunction(y)
        grad = self.neural_network.calcGradients(y)
        return cost, grad

    def train(self, X, y):
        params0 = self.neural_network.getWeights()

        options = {'maxiter': 2000, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',
                                 args=(X, y), options=options, callback=self.callbackFunction)

        self.neural_network.setWeights(_res.x)
        self.optimizationResults = _res