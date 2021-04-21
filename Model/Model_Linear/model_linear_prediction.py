import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np

DIR = "Model/saves/";
#DIR_MODEL = DIR + "model1.pth";
#DIR_OPTIM = DIR + "model1_optimiser.pth"
DIR_WEIGHTS = DIR + "model1_weights.pth";


# Define Model (for pickle import)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__();
        self.Flatten = nn.Flatten();    # Convert images from 2D to 1D array

        l1 = 25*25;
        l2 = 22*22;
        l3 = 18*18;
        l4 = 13*13;
        l5 = 4*4;

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
    
    #Close previous plot if it's still open
    plt.close()

    #Get array of indices 
    index = np.arange(len(probability))

    #Plot index on x-axis and probability on y-axis
    plt.bar(index, probability)

    #Add labels
    plt.xlabel('Digit', fontsize=15)
    plt.ylabel('Probability', fontsize=20)
    plt.xticks(index, fontsize=8, rotation=30)
    plt.title('Model Prediction Probability')
    plt.show()

def predict(img, weights_dir):
    try:
        trans = transforms.ToTensor()
        model = Model()
        model.load_state_dict(torch.load(weights_dir), strict= False)
        # strict=False; so the dict is loaded correctly despite .module labelling from the nn.Sequential() structure
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

        return int(pred), plt;
    except Exception as e:
        print(e);
