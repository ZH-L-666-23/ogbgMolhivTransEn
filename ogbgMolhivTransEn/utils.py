import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            y_true.append(data.y.view(-1).cpu())
            y_pred.append(out.view(-1).cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    # 计算ROC-AUC
    auc = roc_auc_score(y_true, y_pred)
    return auc


def plot_metrics(train_losses, val_aucs, config):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label='Validation AUC')
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, 'training_metrics.png'))
    plt.close()