from tqdm import tqdm

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def train(model, criterion, trainloader, optimizer, device, epoch):
    print('\nTrain epoch: %d' % epoch)
    
    model.train()
    pbar = tqdm(trainloader)

    losses = []
    accuracies = []

    total_loss = 0
    correct = 0
    total = 0
    processed = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        y_preds = model(inputs)

        loss = criterion(y_preds, targets)

        _loss = loss.data.cpu().numpy().item()
        total_loss += _loss
        losses.append(_loss)

        loss.backward()
        optimizer.step()

        
        pred = y_preds.argmax(dim=1, keepdim=True)

        total += targets.size(0)
        correct += pred.eq(targets.view_as(pred)).sum().item()
        processed += len(inputs)

        pbar.set_description(desc= f'Loss={loss.item()} batch={batch_idx} LR={get_lr(optimizer):0.5f} Accuracy={100*correct/processed:0.2f}')
        accuracies.append(100. * correct / processed)

        if batch_idx == 10:
            break

    return losses, accuracies
