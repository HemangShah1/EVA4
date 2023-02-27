import os
from tqdm import tqdm
import torch

def test(model, criterion, testloader, device, best_acc, epoch):
    print('\nTest epoch: %d' % epoch)
    model.eval()
    
    pbar = tqdm(testloader)

    losses = []
    accuracies = []

    total_loss = 0
    correct = 0
    total = 0
    processed = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            y_preds = model(inputs)
            
            loss = criterion(y_preds, targets)

            _loss = loss.data.cpu().numpy().item()
            total_loss += _loss
            losses.append(_loss)

            pred = y_preds.argmax(dim=1, keepdim=True)
            total += targets.size(0)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            processed += len(inputs)

            pbar.set_description(desc= f'Loss={loss.item()} batch={batch_idx} Accuracy={100*correct/processed:0.2f}')
            

    total_loss /= len(testloader.dataset)
    losses.append(total_loss)
    
    accuracy = 100. * correct / len(testloader.dataset)
    accuracies.append(accuracy)

    # Save checkpoint.
    if accuracy > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = accuracy

    return losses, accuracies, best_acc
