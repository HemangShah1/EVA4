import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms


import os
import argparse
from tqdm import tqdm

from models import *
import datasets
from utils import train, test, util



def create_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return train_transforms, test_transforms


def run(device, model, lr, trainloader, testloader, num_epochs):
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.parallel.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    start_epoch = 0
    best_acc = 0
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    for epoch in range(start_epoch, start_epoch + num_epochs):
        losses, accuracies = train.train(model, criterion, trainloader, optimizer, device, epoch)
        train_losses.extend(losses)
        train_accuracies.extend(accuracies)

        losses, accuracies, best_acc = test.test(model, criterion, testloader, device, best_acc, epoch)
        test_losses.extend(losses)
        test_accuracies.extend(accuracies)

        scheduler.step()

    util.plot_metrics(train_accuracies, train_losses, test_accuracies, test_losses)
    



if __name__ == '__main__':


    train_transforms, test_transforms = create_transforms()
    trainloader, testloader, classes, trainset, testset, stats = datasets.get_CIFAR10_datasets(train_transforms, test_transforms)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    model = ResNet18()
    num_epochs = 1
    lr = 0.01

    run(device, model, lr, trainloader, testloader, num_epochs)

    wrong_predictions = util.get_wrong_predictions(model, testloader, device)
    util.plot_misclassified(wrong_predictions, stats['mean'], stats['std'], 10, classes)
