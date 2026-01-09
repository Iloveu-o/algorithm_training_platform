# -*- coding: utf-8 -*-
# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import functions as func
import os
import matplotlib.pyplot as plt

# ---------- 载入配置 ----------
# 加载预先配置的超参数设置
settings = torch.load('./Settings/settings_SoH_CaseA.pth')
# 设置输入序列长度
seq_len = 1
# 设置验证集的比例
perc_val = 0.2
# 设置实验重复次数
num_rounds = 1
# 从设置中获取批次大小
batch_size = settings['batch_size']
# 从设置中获取训练轮数
num_epoch = settings['num_epoch']
# 从设置中获取LSTM层数（取第一个值作为整数）
num_layers = settings['num_layers'][0]  # 注意这里是整数
# 从设置中获取隐藏层维度
hidden_dim = settings['num_neurons'][0]
# 从设置中获取动态输入库
inputs_lib_dynamical = settings['inputs_lib_dynamical']
# 从设置中获取动态输入维度库
inputs_dim_lib_dynamical = settings['inputs_dim_lib_dynamical']

# 检测是否有可用的GPU设备，如果有则使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 载入数据 ----------
# 加载Severson电池数据集
addr = './SeversonBattery.mat'
data = func.SeversonBattery(addr, seq_len=seq_len)

# ---------- 定义 BiLSTM 模型 ----------
class BiLSTMModel(nn.Module):
    """
    双向LSTM模型用于电池健康状态（SoH）预测。
    
    参数:
        input_dim: 输入特征的维度
        hidden_dim: LSTM隐藏层的维度
        output_dim: 输出的维度（通常为1，预测SoH）
        num_layers: LSTM的层数
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        # 调用父类构造函数
        super(BiLSTMModel, self).__init__()
        # 创建双向LSTM层
        # bidirectional=True表示使用双向LSTM，输出维度会是hidden_dim*2
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        # 创建全连接层，将LSTM输出映射到最终预测值
        # 输入维度是hidden_dim*2，因为是双向LSTM
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过LSTM层进行前向传播
        # out的形状为(batch_size, seq_len, hidden_dim*2)
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出并通过全连接层
        # out[:, -1, :]表示取每个样本的最后一个时间步
        out = self.fc(out[:, -1, :])
        # 增加一个维度以匹配目标张量的形状
        return out.unsqueeze(1)

# ---------- 初始化评估指标 ----------
# 初始化用于存储不同输入配置下评估指标平均值的字典
metric_mean = {'train': np.zeros((len(inputs_lib_dynamical), 1)),
               'val': np.zeros((len(inputs_lib_dynamical), 1)),
               'test': np.zeros((len(inputs_lib_dynamical), 1))}

# 初始化用于存储不同输入配置下评估指标标准差的字典
metric_std = {'train': np.zeros((len(inputs_lib_dynamical), 1)),
              'val': np.zeros((len(inputs_lib_dynamical), 1)),
              'test': np.zeros((len(inputs_lib_dynamical), 1))}

# 初始化结果字典
results = {}

# ---------- 训练 & 测试 ----------
# 遍历所有动态输入配置
for l in range(len(inputs_lib_dynamical)):
    # 获取当前配置的输入和维度
    inputs_dynamical, inputs_dim_dynamical = inputs_lib_dynamical[l], inputs_dim_lib_dynamical[l]
    # 初始化用于存储每一轮结果的字典
    metric_rounds = {'train': np.zeros(num_rounds),
                     'val': np.zeros(num_rounds),
                     'test': np.zeros(num_rounds)}

    # 进行多轮实验
    for round in range(num_rounds):
        # 创建数据分割，选择训练单元[91, 124]和测试单元[100]
        inputs_dict, targets_dict = func.create_chosen_cells(
            data,
            idx_cells_train=[91, 124],
            idx_cells_test=[100],
            perc_val=perc_val
        )
        # 将训练输入移到指定设备（GPU或CPU）
        inputs_train = inputs_dict['train'].to(device)
        # 将验证输入移到指定设备
        inputs_val = inputs_dict['val'].to(device)
        # 将测试输入移到指定设备
        inputs_test = inputs_dict['test'].to(device)
        # 将训练目标的第一列（容量衰减）移到指定设备
        targets_train = targets_dict['train'][:, :, 0:1].to(device)
        # 将验证目标移到指定设备
        targets_val = targets_dict['val'][:, :, 0:1].to(device)
        # 将测试目标移到指定设备
        targets_test = targets_dict['test'][:, :, 0:1].to(device)

        # 禁用cuDNN以避免版本不兼容问题
        torch.backends.cudnn.enabled = False
        # 创建BiLSTM模型实例并移到指定设备
        model = BiLSTMModel(inputs_train.shape[2], hidden_dim, 1, num_layers).to(device)

        # 创建Adam优化器
        optimizer = optim.Adam(model.parameters(), lr=settings['lr'])
        # 创建学习率调度器（每step_size个epoch后乘以gamma）
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=settings['step_size'], gamma=settings['gamma'])
        # 创建均方误差损失函数
        criterion = nn.MSELoss()

        # 创建训练数据集
        train_set = TensorDataset(inputs_train, targets_train)
        # 创建训练数据加载器
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        # 开始训练循环
        for epoch in range(num_epoch):
            # 设置模型为训练模式
            model.train()
            # 遍历训练数据加载器中的每个批次
            for x_batch, y_batch in train_loader:
                # 清零优化器梯度
                optimizer.zero_grad()
                # 前向传播：获取模型预测结果
                output = model(x_batch)
                # 计算损失
                loss = criterion(output, y_batch)
                # 反向传播计算梯度
                loss.backward()
                # 更新模型参数
                optimizer.step()
            # 更新学习率
            scheduler.step()

        # --------- 测试 ----------
        # 设置模型为评估模式
        model.eval()

        # 定义评估指标计算函数
        def eval_metrics(x, y):
            """
            计算模型评估指标。
            
            参数:
                x: 输入数据
                y: 目标数据
                
            返回:
                metrics: 包含各种评估指标的字典
                pred: 预测的SoH值
                y: 真实的SoH值
            """
            with torch.no_grad():
                # 获取模型预测结果
                pred = model(x)
                # 将容量损失转换为健康状态（SoH）
                pred = 1. - pred
                y = 1. - y

                # 计算平均绝对误差（MAE）
                mae = torch.mean(torch.abs(pred - y)).item()
                # 计算均方误差（MSE）
                mse = torch.mean((pred - y) ** 2).item()
                # 计算误差平方和（SSE）
                sse = torch.sum((pred - y) ** 2).item()
                # 计算相对均方根百分比误差（RMSPE）
                rmspe = torch.sqrt(torch.mean(((pred - y) / y) ** 2)).item()

                # 计算R²分数
                # R^2 = 1 - SS_res / SS_tot
                ss_res = torch.sum((y - pred) ** 2)
                ss_tot = torch.sum((y - torch.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot
                r2 = r2.item()

                return {'MAE': mae, 'MSE': mse, 'SSE': sse, 'RMSPE': rmspe, 'R2': r2}, pred, y


        # 调用评价函数计算各数据集上的指标
        # 在训练集上评估
        metrics_train, _, _ = eval_metrics(inputs_train, targets_train)
        # 在验证集上评估
        metrics_val, _, _ = eval_metrics(inputs_val, targets_val)
        # 在测试集上评估
        metrics_test, pred_test, true_test = eval_metrics(inputs_test, targets_test)

        # 将当前轮次的RMSPE指标保存
        metric_rounds['train'][round] = metrics_train['RMSPE']
        metric_rounds['val'][round] = metrics_val['RMSPE']
        metric_rounds['test'][round] = metrics_test['RMSPE']

        # 在第一轮实验时保存结果
        if round == 0:
            # 更新结果字典
            results['U_pred'] = pred_test.cpu().numpy().squeeze()  # 预测的SoH值
            results['U_true'] = true_test.cpu().numpy().squeeze()  # 真实的SoH值
            results['Cycles'] = inputs_test[:, :, -1:].cpu().numpy().squeeze()  # 循环次数
            results['metrics_test'] = metrics_test  # 保存所有测试指标

            # 只打印一次指标（在第一轮时）
            m = results['metrics_test']
            print(f"Test Evaluation Metrics:")
            print(f"MAE   : {m['MAE']:.6f}")  # 平均绝对误差
            print(f"MSE   : {m['MSE']:.6f}")  # 均方误差
            print(f"SSE   : {m['SSE']:.6f}")  # 误差平方和
            print(f"RMSPE : {m['RMSPE']:.6f}")  # 相对均方根百分比误差
            print(f"R2    : {m['R2']:.6f}")  # R²分数




    # 计算该输入配置下的平均指标
    metric_mean['train'][l, 0] = np.mean(metric_rounds['train'])
    metric_mean['val'][l, 0] = np.mean(metric_rounds['val'])
    metric_mean['test'][l, 0] = np.mean(metric_rounds['test'])
    # 计算该输入配置下的标准差
    metric_std['train'][l, 0] = np.std(metric_rounds['train'])
    metric_std['val'][l, 0] = np.std(metric_rounds['val'])
    metric_std['test'][l, 0] = np.std(metric_rounds['test'])

# ---------- 保存结果 ----------
# 将平均指标和标准差添加到结果字典中
results['metric_mean'] = metric_mean
results['metric_std'] = metric_std
# 保存结果到文件
torch.save(results, './results/SoH_CaseA_BiLSTM.pth')

# ---------- 绘图 ----------
# 创建图形窗口
plt.figure(figsize=(4, 3))
# 绘制真实的SoH值
plt.plot(results['Cycles'], results['U_true'], label='True SOH')
# 绘制预测的SoH值（使用虚线）
plt.plot(results['Cycles'], results['U_pred'], label='Predicted SOH', linestyle='--')
# 设置x轴标签
plt.xlabel('Cycle')
# 设置y轴标签
plt.ylabel('SOH')
# 显示图例
plt.legend()
# 自动调整布局
plt.tight_layout()
# 保存图形到文件
plt.savefig('BiLSTM_SoH_Result.png', dpi=300)
# 显示图形
plt.show()