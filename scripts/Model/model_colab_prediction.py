import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np

#Network class 
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__();
        self.Flatten = nn.Flatten();    # Convert images from 2D to 1D array

        l1 = 26*26;
        l2 = 24*24;
        l3 = 20*20;
        l4 = 18*18;
        l5 = 16*16;
        l6 = 14*14;
        l7 = 12*12;

        l8 = 10*10;
        l9 = 8*8;
        l10 = 6*6;

        self.linear_relu_stack = nn.Sequential(
            # 6 layer stack
            # linear downscaling in data size
            # ReLu to identify non-linear behaviour for closer fitting
            nn.Linear(28*28, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, l3),
            nn.ReLU(),
            nn.Linear(l3, l4),
            nn.ReLU(),
            nn.Linear(l4, l5),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.Flatten(x);
        logits = self.linear_relu_stack(x);
        return logits;



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
    model = Model()
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
