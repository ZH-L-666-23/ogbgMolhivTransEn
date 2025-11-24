import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, Constant


class GraphDataset:
    def __init__(self, config):
        self.config = config
        self.dataset = PygGraphPropPredDataset(
            name=config.dataset_name,
            root=config.data_dir,
            transform=Compose([Constant(value=1)])  # 添加常数特征
        )
        self.split_idx = self.dataset.get_idx_split()

        # 添加虚拟节点特征（如果不存在）
        if self.dataset[0].x is None:
            for data in self.dataset:
                data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)

    def get_loaders(self):
        train_loader = DataLoader(
            self.dataset[self.split_idx["train"]],
            batch_size=self.config.batch_size,
            shuffle=True
        )
        valid_loader = DataLoader(
            self.dataset[self.split_idx["valid"]],
            batch_size=self.config.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            self.dataset[self.split_idx["test"]],
            batch_size=self.config.batch_size,
            shuffle=False
        )
        return train_loader, valid_loader, test_loader

    @property
    def num_features(self):
        return self.dataset[0].x.size(1)

    @property
    def num_classes(self):
        return self.dataset.num_tasks