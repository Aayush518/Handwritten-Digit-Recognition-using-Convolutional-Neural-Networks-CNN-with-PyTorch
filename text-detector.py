import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

use_cuda = torch.cuda.is_available()

transform = transforms.ToTensor()

# load the training and test datasets
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Create training and test dataloaders

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 16

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 800, 5)
        self.conv4 = nn.Conv2d(800, 800, 1)
        self.conv5 = nn.Conv2d(800, 10, 1)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)

        return x


model = Model()
if use_cuda:
    model = model.cuda()


def train_model(model, train_loader, test_loader, n_epochs):
    """helper function to train the model
        inputs: model --> our model
                train_loader --> the train images loader
                test_loader --> the test images loader
                n_epochs --> number of iterations
    """
    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()
    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    min_test_accuracy = 0
    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        train_accuracy = 0
        test_accuracy = 0
        ###################
        # train the model #
        ###################
        model.train()
        for images, labels in train_loader:
            # move tensors to GPU if CUDA is available
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            logits = model(images).squeeze()
            _, outs_k = logits.topk(1, dim=1)
            equals = (outs_k == labels.view(*outs_k.shape))
            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # calculate the batch loss
            loss = criterion(logits, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * images.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                # move tensors to GPU if CUDA is available
                if use_cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                logits = model(images).squeeze()
                _, outs_k = logits.topk(1, dim=1)
                equals = (outs_k == labels.view(*outs_k.shape))
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        train_accuracy = train_accuracy / len(train_loader)
        test_accuracy = test_accuracy / len(test_loader)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tTrain Accuracy: {:.6f} \tTest Accuracy: {:.6f}'.format(
            epoch, train_loss, train_accuracy, test_accuracy))

        # save the model if test accuracy has increased
        if test_accuracy > min_test_accuracy:
            print('Test accuracy increased ({:.6f} --> {:.6f}). Saving model...'.format(
                min_test_accuracy, test_accuracy))
            torch.save(model.state_dict(), 'model.pt')
            min_test_accuracy = test_accuracy


# Set the number of epochs to train the model
n_epochs = 10

# Call the train_model function
train_model(model, train_loader, test_loader, n_epochs)
