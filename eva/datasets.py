import torch.utils.data
import torchvision

def get_CIFAR10_datasets(train_transforms, test_transforms):
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
    
    mean = trainset.data.mean(axis=(0,1,2))/255
    std = trainset.data.std(axis=(0,1,2))/255
    stats = {'mean': mean, 'std': std}

    return trainloader, testloader, classes, trainset, testset, stats
