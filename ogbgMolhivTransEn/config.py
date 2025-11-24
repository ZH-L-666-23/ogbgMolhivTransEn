import torch
class Config:
    def __init__(self):
        # 数据集配置
        self.dataset_name = 'ogbg-molhiv'
        self.data_dir = './data'

        # 模型参数
        self.gnn_hidden_dim = 128
        self.gnn_num_layers = 5
        self.gnn_dropout = 0.2
        self.transformer_hidden_dim = 128
        self.transformer_num_heads = 8
        self.transformer_num_layers = 4
        self.transformer_dropout = 0.1

        # 训练参数
        self.batch_size = 128
        self.epochs = 100
        self.lr = 0.001
        self.weight_decay = 1e-5
        self.patience = 20

        # 设备配置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 日志和保存
        self.log_interval = 10
        self.save_dir = './checkpoints'

    def __str__(self):
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])


if __name__ == '__main__':
    config = Config()
    print(config)