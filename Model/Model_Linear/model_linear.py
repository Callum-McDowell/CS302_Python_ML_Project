# Callum McDowell - 8th April 2021

# First custom model.
# [TODO]: Explain further, add timer, tweak: stacks + loss fn + optimiser
# https://pytorch.org/tutorials/beginner/basics/intro.html


#====== Libraries ======#
import resources as r
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os;
import torch;
import torchvision;
import time;
# Data:
from torch.utils import data;
from torchvision import datasets, models;
from torchvision.transforms import ToTensor, Lambda;
# Model:
from torch import nn, cuda;

MODEL_NAME = "Linear"


#====== Hyper Parameters ======#
number_of_epochs = 4; # arbitrary
batch_size = 64;        # arbitrary
learning_rate = 1e-1;   # arbitrary
        
#====== Model ======#
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
# trainModel()
# 

class modelTrainingFramework():
    # Wrapper class so that functions can be called on instantiated object
    def trainModel(self):
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

        #====== Model Instance ======#
        device = "cpu";
        if cuda.is_available():
            device = "cuda";

        self.net = Model().to(device);

        #====== Loss and Optimiser ======#
        self.loss_fn = nn.CrossEntropyLoss()
            # NOTE: try NLLLoss or CrossEntropyLoss for negative log
            #       (negative log is better for classification)
        self.optimiser = torch.optim.SGD(self.net.parameters(), lr= learning_rate);
            # NOTE: experiment with different optimisers, not just SDG

        #====== Training Epochs ======#
        print(f"Starting training with {MODEL_NAME} on device {device}\n{'=' * 24}");

        t0 = time.perf_counter()

        for i in range(number_of_epochs):
            print(f"Epoch {i+1}\n----------------------------")
            self.train(loader_trainset, self.net, self.loss_fn, self.optimiser);
            accuracy = self.test(loader_testset, self.net, self.loss_fn);

        t1 = time.perf_counter();
        print(f"Finished in {(t1 - t0):>.2f}s.");
        print("FIN.")

        return accuracy;



    def train(self, dataloader, model, loss_fn, optimiser):
        # t0 = time.perf_counter();

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
                
                torch.save(model, "Model/saves/model1.pkl");
                torch.save(optimiser.state_dict(), "Model/saves/model1_optimiser.pkl");
                torch.save(model.state_dict(), "Model/saves/model1_weights.pkl");

                #self.completion = (100 * index) / len(dataloader); #????? TODO
                #progressBar.setValue(self.completion);

                # t1 = time.perf_counter();
                # print(f"Training epoch took {(t1 - t0):>.2f}s.");
                # t0 = timer.perf_counter();


    def test(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset);
        test_loss, correct = 0, 0;

        with torch.no_grad():       # disable learning; no back propagation
            for X, y in dataloader:
                pred = model(X);
                test_loss += loss_fn(pred, y).item();
                correct += (pred.argmax(1) == y).type(torch.float).sum().item();
            
            test_loss /= size;  # equivalent to test_loss = test_loss/size;
            correct /= size;
            accuracy = 100*correct;
            print (f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n");

        return accuracy;
