import torchvision
import torchvision.datasets as datasets

def importDataset():
    mnist_trainset = datasets.MNIST(root='Dataset/trainset', train=True, download=False, transform=None)
    mnist_testset = datasets.MNIST(root='Dataset/testset', train=False, download=False, transform=None)

    return mnist_trainset, mnist_testset

trainset, testset = importDataset()
