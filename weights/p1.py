import os

gamma = 0.9  # 折扣因子
epsilon_beg = 1.0  # 初始探索率
epsilon_end = 0.2  # 最小探索率
epsilon_dec = 0.999  # 探索率衰减率
update_cycle = 20  # 更新周期
lr = 1e-4  # 学习率
batch_size = 64  # 批大小
episodes = 50000  # 训练次数
capacity = 5000  # 记忆库容量
hidden_size = 128  # 隐藏层大小
weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w1.pth")  # 权重保存路径
