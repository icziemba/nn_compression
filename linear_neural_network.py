import numpy as np
import torch
import torch.nn as nn

class LinearNeuralNetwork(nn.Module):
    MNIST_ARCH = [28*28, 10, 10]

    def __init__(self, architecture=MNIST_ARCH, seed=None):
        super(LinearNeuralNetwork, self).__init__()
        
        # Setup the layers
        tmp = []
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
            
            tmp.append(linear)
        
        # Need to use module list in order to generate parameters
        self.layers = nn.ModuleList(tmp)
    
    def forward(self, x):
        out = x        
        for i, l in enumerate(self.layers):
            out = l(out)
        return out
