# Mazen Darwish

#This file contains the basic functions for predicting canvas input using the models

import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np

#Network class 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

#This function plots the probability bar graph
def plot_bar(probability):
    #Get array of indices 
    index = np.arange(len(probability))

    #Plot index on x-axis and probability on y-axis
    plt.bar(index, probability)

    #Add labels
    plt.xlabel('Digit', fontsize=15)
    plt.ylabel('Probability', fontsize=20)
    plt.xticks(index, fontsize=8, rotation=30)
    plt.title('Model Prediction Probability')
    return plt

def predict(img, modelFilename):
    trans = transforms.ToTensor()
    model = Net()
    model.load_state_dict(torch.load(modelFilename))
    output = model(trans(img))
    pred = output.data.max(1, keepdim=True)[1]

    #Getting the relative probability of the predictions
    relative_probability = output[0].tolist()
    if min(relative_probability) < 0:
        for value in relative_probability:
            ind = relative_probability.index(value)
            relative_probability[ind] = value + (-(min(output[0].tolist())))

    #Plot probability graph
    plt = plot_bar(relative_probability)
    return int(pred), plt
