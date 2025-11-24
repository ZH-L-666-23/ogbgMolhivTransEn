import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os

from config import Config
from dataset import GraphDataset
from model import GNNTransformer
from utils import create_dir, save_model, load_model, evaluate, plot_metrics


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        y = data.y.view(-1).float()
        loss = criterion(out.view(-1), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


def main():
    # 初始化配置
    config = Config()
    print("Configuration:")
    print(config)

    # 创建保存目录
    create_dir(config.save_dir)

    # 加载数据集
    dataset = GraphDataset(config)
    train_loader, valid_loader, test_loader = dataset.get_loaders()

    print(f"Dataset: {config.dataset_name}")
    print(f"Number of training graphs: {len(train_loader.dataset)}")
    print(f"Number of validation graphs: {len(valid_loader.dataset)}")
    print(f"Number of test graphs: {len(test_loader.dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")

    # 初始化模型
    model = GNNTransformer(
        config,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes
    ).to(config.device)

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 训练变量
    best_val_auc = 0
    best_epoch = 0
    train_losses = []
    val_aucs = []

    # 训练循环
    print("Starting training...")
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # 训练
        loss = train(model, train_loader, optimizer, criterion, config.device)
        train_losses.append(loss)

        # 验证
        val_auc = evaluate(model, valid_loader, config.device)
        val_aucs.append(val_auc)

        # 记录时间
        epoch_time = time.time() - epoch_start

        # 打印进度
        if epoch % config.log_interval == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Time: {epoch_time:.2f}s")

        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            save_model(model, optimizer, epoch, os.path.join(config.save_dir, 'best_model.pth'))
            print(f"New best model at epoch {epoch} with Val AUC: {val_auc:.4f}")

        # 提前停止
        if epoch - best_epoch >= config.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 训练结束
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds")
    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    # 绘制训练曲线
    plot_metrics(train_losses, val_aucs, config)

    # 加载最佳模型并在测试集上评估
    print("Evaluating on test set...")
    load_model(model, optimizer, os.path.join(config.save_dir, 'best_model.pth'))
    test_auc = evaluate(model, test_loader, config.device)
    print(f"Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()


