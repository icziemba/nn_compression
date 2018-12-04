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

    csv_weights = "student_model_weights.csv"
    with open(csv_weights, mode="w") as fd:
        csv = csv.writer(fd, delimiter=',')


        for i in range(len(weights)):
            numpy_weights = weights[i].cpu().detach().numpy()
            numpy_biases = biases[i].cpu().detach().numpy()

            csv.writerow(["Weights between layer {} and {}".format(i, i + 1)])
            for row in range(len(numpy_weights)):
                weight_list = list(numpy_weights[row])
                csv.writerow(weight_list)

            csv.writerow(["Biases between layer {} and {}".format(i, i + 1)])
            for row in range(len(numpy_biases)):
                bias = numpy_biases[row]
                csv.writerow([bias])

        csv.writerow([""])
        csv.writerow([""])

    print("Weights file: {}".format(csv_weights))
