import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import convolutional_neural_network as cnn
import linear_neural_network as lnn
import distillation_loss as dl
import os
import csv

if __name__ == "__main__":
    student_save = "student_layers_1_units_10_alpha_1_temp_2.0_epoch_10_lr_0.01"
    student_model = torch.load(student_save)

    weights = student_model.weights()
    biases = student_model.biases()

    max_weight_value = weights[0].max()
    min_weight_value = weights[0].min()
    for i in range(1, len(weights)):
        next_value = weights[1].max()
        if next_value > max_weight_value:
            max_weight_value = next_value

        if next_value < min_weight_value:
            min_weight_value = next_value

    max_bias_value = biases[0].max()
    min_bias_value = biases[0].min()
    for i in range(1, len(biases)):
        next_value = biases[1].max()
        if next_value > max_bias_value:
            max_bias_value = next_value

        if next_value < min_weight_value:
            min_bias_value = next_value

    print("Model {}: Max weight {} Min weight {} Max bias {} Min bias{}".format(
        student_save, max_weight_value, min_weight_value, max_bias_value, min_bias_value))
