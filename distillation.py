import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import convolutional_neural_network as cnn
import linear_neural_network as lnn
import distillation_loss as dl
import os

# Accuracy function
def prediction_accuracy(predicted, actual):
    return (actual==predicted).sum().item()/len(actual)

# Raw logits to class prediction
def logits_to_prediction(logits):
    return torch.argmax(logits.data, dim=1)

def train(model, train_loader, device, comp_model=None, flat_data=False, lr=0.001, num_epochs=5, loss_func=nn.CrossEntropyLoss()):
    num_batches = len(train_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if flat_data:
                comp_images = images.to(device)
                images = images.view(train_loader.batch_size, -1).to(device)
            else:
                images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)

            if (comp_model and flat_data):
                comp_outputs = comp_model(comp_images)
                loss = loss_func(outputs, comp_outputs, labels)
            else:
                loss = loss_func(outputs, labels)
        
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('epoch [{}/{}], batch [{}/{}], loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, num_batches, loss.item()))
    return


def accuracy(model, test_loader, flat_data=False):
    batch_accuracy = []
    for i, (images, labels) in enumerate(test_loader):
        if flat_data:
            images = images.view(test_loader.batch_size, -1).to(device)
        else:
            images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        prediction = logits_to_prediction(outputs)
        accuracy = prediction_accuracy(prediction, labels)
        batch_accuracy.append(accuracy)

    return sum(batch_accuracy) / len(batch_accuracy)

if __name__ == "__main__":
    # Teacher model save location
    teacher_save = os.getcwd() + "/teacher_model"

    # Device to be used
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # The data sets
    train_dataset = torchvision.datasets.MNIST(root='./MNIST',
                                               train=True, 
                                               transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='./MNIST',
                                              train=False, 
                                              transform=transforms.ToTensor())

    batch_size = 100  # how many examples are processed at each step
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              shuffle=False)

    # Train the teacher model
    if os.path.isfile(teacher_save):
        print("Loading teacher model: " + teacher_save)
        teacher = torch.load(teacher_save)
    else:
        print("Training teacher model...")
        teacher = cnn.ConvNeuralNetwork().to(device)
        train(teacher, train_loader, device)
        torch.save(teacher, teacher_save)
        print("Saving teacher model: " + teacher_save)

    # Get the teacher accuracy
    teacher_accuracy = accuracy(teacher, test_loader)
    print("Teacher accuracy: {}".format(teacher_accuracy))

    # Train the student model
    student = lnn.LinearNeuralNetwork().to(device)
    train(student, train_loader, device, comp_model=teacher, flat_data=True, loss_func=dl.DistillationLoss(alpha=0.4, temp=1.0))

    # Get the student accuracy
    student_accuracy = accuracy(student, test_loader, flat_data=True)
    print("Student accuracy: {}".format(student_accuracy))
