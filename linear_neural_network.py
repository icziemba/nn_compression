import numpy as np
import torch
import torch.nn as nn

class LinearNeuralNetwork(nn.Module):
    MNIST_ARCH = [28*28, 10, 10]

    def __init__(self, architecture=MNIST_ARCH, seed=None):
        super(LinearNeuralNetwork, self).__init__()
        
        # Setup the layers
        self.models = []
        for i in range(len(architecture) - 1):
            inputs = architecture[i]
            outputs = architecture[i + 1]

            # Linear model for connecting neurons
            linear = nn.Linear(inputs, outputs)

            # Initialize weights and bias (similar to class)
            np.random.seed(seed=seed)
            weights = np.random.randn(architecture[i + 1], architecture[i])
            bias = np.random.randn(architecture[i + 1])
            
            linear.weight.data = torch.from_numpy(weights).float()
            linear.bias.data = torch.from_numpy(bias).float()
            
            self.models.append(linear)
        
        # Need to use module list in order to generate parameters
        self.layers = nn.ModuleList(self.models)
    
    def forward(self, x):
        out = x        
        for i, l in enumerate(self.layers):
            out = l(out)
        return out

    def weights(self):
        weights = []
        for model in self.models:
            weights.append(model.weight)
        return weights

    def biases(self):
        biases = []
        for model in self.models:
            biases.append(model.bias)
        return biases

    def max_weight(self):
        weights = self.weights()

        max_value = weights[0].max()
        for i in range(1, len(weights)):
            next_value = weights[1].max()
            if next_value > max_value:
                max_values = next_value

        return max_value.item()

    def max_bias(self):
        biases = self.biases()

        max_value = biases[0].max()
        for i in range(1, len(biases)):
            next_value = biases[1].max()
            if next_value > max_value:
                max_values = next_value

        return max_value.item()
