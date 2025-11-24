import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GNNTransformer(nn.Module):
    def __init__(self, config, num_features, num_classes):
        super().__init__()
        self.config = config
        self.num_features = num_features
        self.num_classes = num_classes

        # GNN部分
        self.gnn_convs = nn.ModuleList()
        self.gnn_bns = nn.ModuleList()

        # 输入层
        self.gnn_convs.append(GCNConv(num_features, config.gnn_hidden_dim))
        self.gnn_bns.append(nn.BatchNorm1d(config.gnn_hidden_dim))

        # 隐藏层
        for _ in range(config.gnn_num_layers - 1):
            self.gnn_convs.append(GCNConv(config.gnn_hidden_dim, config.gnn_hidden_dim))
            self.gnn_bns.append(nn.BatchNorm1d(config.gnn_hidden_dim))

        # Transformer部分
        encoder_layer = TransformerEncoderLayer(
            d_model=config.transformer_hidden_dim,
            nhead=config.transformer_num_heads,
            dim_feedforward=config.transformer_hidden_dim * 4,
            dropout=config.transformer_dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, config.transformer_num_layers)

        # 分类器
        self.fc1 = nn.Linear(config.transformer_hidden_dim, config.transformer_hidden_dim)
        self.fc2 = nn.Linear(config.transformer_hidden_dim, num_classes)

        # 位置编码
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 1000, config.transformer_hidden_dim)
        )
        nn.init.xavier_uniform_(self.positional_encoding)

        self.dropout = nn.Dropout(config.gnn_dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 添加这一行，将x转换为float类型
        x = x.float()  # 关键修复

        # GNN处理
        for i in range(len(self.gnn_convs)):
            x = self.gnn_convs[i](x, edge_index)
            x = self.gnn_bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        # 获取每个图的节点数
        unique, counts = torch.unique(batch, return_counts=True)
        max_nodes = counts.max().item()

        # 将节点特征组织为序列
        node_features = []
        for graph_idx in unique:
            graph_mask = (batch == graph_idx)
            graph_nodes = x[graph_mask]
            # 填充到最大节点数
            padded = F.pad(graph_nodes, (0, 0, 0, max_nodes - graph_nodes.size(0)))
            node_features.append(padded)

        node_features = torch.stack(node_features)  # [batch_size, max_nodes, hidden_dim]

        # 位置编码
        pos_enc = self.positional_encoding[:, :max_nodes, :]
        node_features = node_features + pos_enc

        # Transformer处理
        transformer_out = self.transformer_encoder(node_features)

        # 全局平均池化
        graph_emb = transformer_out.mean(dim=1)

        # 分类器
        graph_emb = F.relu(self.fc1(graph_emb))
        graph_emb = self.dropout(graph_emb)
        out = self.fc2(graph_emb)

        return out