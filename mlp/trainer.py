from scipy import optimize


class Trainer:
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
