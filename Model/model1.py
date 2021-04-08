# Callum McDowell - 8th April 2021

# First custom model.
# [TODO]: Explain further, add timer, tweak: stacks + loss fn + optimiser
# https://pytorch.org/tutorials/beginner/basics/intro.html


#====== Libraries ======#
import os;
import torch;
import torchvision;
# Data:
from torch.utils import data;
from torchvision import datasets, models;
from torchvision.transforms import ToTensor, Lambda;
# Model:
from torch import nn, cuda;



#====== Hyper Parameters ======#
number_of_epochs = 4; # arbitrary
batch_size = 64;        # arbitrary
learning_rate = 1e-1;   # arbitrary

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

        self.linear_relu_stack = nn.Sequential(
            # 6 layer stack
            # linear downscaling in data size
            # ReLu to identify non-linear behaviour for closer fitting
            nn.Linear(28*28, 21*21),
            nn.ReLU(),
            nn.Linear(21*21, 14*14),
            nn.ReLU(),
            nn.Linear(14*14, 16),
            nn.ReLU(),
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
    # NOTE: experiment with different optimisers, not just SDG


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
for i in range(number_of_epochs):
    print(f"Epoch {i+1}\n----------------------------")
    train(loader_trainset, net, loss_fn, optimiser);
    test(loader_testset, net, loss_fn);

# Export and save model
torch.save(net, "Model/saves/model1.pth");
torch.save(optimiser.state_dict(), "Model/saves/model1_optimiser.pth");
torch.save(net.state_dict(), "Model/saves/model1_weights.pth");

print("FIN.")

#accuracy=77.2% at lr=1e-2, batch=64, epoch=10
#accuracy=96.0% at lr=1e-1, batch=64, epoch=4
