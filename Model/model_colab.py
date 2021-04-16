# from google.colab import drive
# drive.mount('/content/drive/')
# SAVE_DIR = "/content/drive/My Drive/Colab Notebooks/"


# Callum McDowell - 8th April 2021

# First custom model.
# [TODO]: Explain further, add timer, tweak: stacks + loss fn + optimiser
# https://pytorch.org/tutorials/beginner/basics/intro.html


#====== Libraries ======#
import os;
import torch;
import torchvision;
import numpy as np;
# Data:
from torch.utils import data;
from torchvision import datasets, models;
from torchvision.transforms import ToTensor, Lambda;
# Model:
from torch import nn, cuda;



#====== Hyper Parameters ======#
number_of_epochs = 10; # arbitrary
batch_size = 64;        # arbitrary
learning_rate = 1e-4;   # arbitrary

device = "cpu";
if cuda.is_available():
    device = "cuda";


#====== Datasets ======#
trainset = datasets.MNIST(
    root="Dataset/trainset",
    train= True,
    download= True,
    transform= ToTensor()
);

loader_trainset = data.DataLoader(
    dataset= trainset,
    batch_size= batch_size,
    shuffle= True,
);

testset = datasets.MNIST(
    root= "Dataset/testset",
    train= False,
    download= True,
    transform= ToTensor()
);

loader_testset = data.DataLoader(
    dataset= testset,
    batch_size= batch_size,
    shuffle= False
);


#====== Model ======#
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
            #nn.Linear(l5, l6),
            #nn.ReLU(),
            # nn.Linear(l6, l7),
            # nn.ReLU(),
 
            # nn.Linear(l7, l8),
            # nn.ReLU(),
            # nn.Linear(l8, l9),
            # nn.ReLU(),
            # nn.Linear(l9, l10),
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.Flatten(x);
        logits = self.linear_relu_stack(x);
        return logits;

net = Model().to(device);

#====== Loss and Optimiser ======#
loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.MSELoss(reduction= 'none');
    # NOTE: try NLLLoss or CrossEntropyLoss for negative log
    #       (negative log is better for classification)
optimiser = torch.optim.SGD(net.parameters(), lr= learning_rate);

optimiser = torch.optim.SGD(
    [
     {"params": net.linear_relu_stack.parameters(), "lr": 1e-1},
     #{"params": net.detail_adjust_stack.parameters(), "lr": 1e-2},
    ]
)


def train(dataloader, model, loss_fn, optimiser):
    # Generic inputs are defined as type
    # Thus we can call generic inherited operations on them
    size = len(dataloader.dataset);
    for index, (X,y) in enumerate(dataloader):
        # as we iterate over '(X,y)' 'index' (from enumerate()) tracks our progress
        
        # Forward
        pred = model(X);
        loss = loss_fn(pred, y);

        # Back propagation:
        optimiser.zero_grad();
        loss.backward();
        optimiser.step();

        if index % 100 == 0:
            loss, progress = loss.item(), index * len(X);
            print(f"loss: {loss:>7f} [{progress:>5d}/{size:>5d}]");
            # e.x. output:  "loss: 1.234567 [    0/60000]""


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset);
    test_loss, correct = 0, 0;

    with torch.no_grad():       # disable learning; no back propagation
        for X, y in dataloader:
            pred = model(X);
            test_loss += loss_fn(pred, y).item();
            correct += (pred.argmax(1) == y).type(torch.float).sum().item();
        
        test_loss /= size;  # equivalent to test_loss = test_loss/size;
        correct /= size;
        print (f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n");


#====== Run Model ======#
import time;
t0 = time.perf_counter();

for i in range(number_of_epochs):
    print(f"Epoch {i+1}\n----------------------------")
    train(loader_trainset, net, loss_fn, optimiser);
    test(loader_testset, net, loss_fn);

# Export and save model
torch.save(net, SAVE_DIR + "model.pth");
torch.save(optimiser.state_dict(), SAVE_DIR + "model_optimiser.pkl");
torch.save(net.state_dict(), SAVE_DIR + "model_weights.pkl");

t1 = time.perf_counter();
print(f"Finished in {(t1 - t0):>.2f}s.");
print("FIN.")

#accuracy=77.2% at lr=1e-2, batch=64, epoch=10
#accuracy=96.0% at lr=1e-1, batch=64, epoch=4
