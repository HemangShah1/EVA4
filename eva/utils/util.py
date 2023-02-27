import torch
import matplotlib.pyplot as plt
import yaml
import numpy as np

def get_wrong_predictions(model, test_loader, device):
    wrong_images=[]
    wrong_labels=[]
    correct_labels=[]
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y_preds = model(data)        
            pred = y_preds.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

            print(f'data.shape: {data.shape}')
            wrong_pred = (pred.eq(target.view_as(pred)) == False)
            print(f'wrong_pred size: {wrong_pred.shape}')
            print(f'data[wrong_pred] size: {data[wrong_pred].shape}')
            
            wrong_images.append(data[wrong_pred])
            wrong_labels.append(pred[wrong_pred])
            correct_labels.append(target.view_as(pred)[wrong_pred])
        
        wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_labels),torch.cat(correct_labels)))
        print(f'Total wrong predictions are {len(wrong_predictions)}')

    return wrong_predictions

def plot_metrics(train_acc, train_losses, test_acc, test_losses):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

    plt.show()


def plot_misclassified(wrong_predictions, mean, std, num_img, classes):
    fig = plt.figure(figsize=(15,12))
    fig.tight_layout()
    for i, (img, pred, correct) in enumerate(wrong_predictions[:num_img]):
        img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
        for j in range(img.shape[0]):
            img[j] = (img[j]*std[j])+mean[j]

        img = np.transpose(img, (1, 2, 0)) 
        ax = fig.add_subplot(5, 5, i+1)
        fig.subplots_adjust(hspace=.5)
        ax.axis('off')

        ax.set_title(f'\nActual : {classes[target.item()]}\nPredicted : {classes[pred.item()]}',fontsize=10)  
        ax.imshow(img)

    plt.show()
