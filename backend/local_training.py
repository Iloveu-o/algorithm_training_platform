# -*- coding: utf-8 -*-
"""
本地模型训练脚本
根据 USER_CONFIG 配置进行 Baseline / BiLSTM / DeepHPM 模型的训练
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import datetime

# 将上级目录加入 path 以导入 functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions as func

# ================= 用户配置区域 =================
USER_CONFIG = {
    # 模型类型: 'Baseline', 'BiLSTM', 'DeepHPM'
    'model_type': 'Baseline',
    
    # 训练参数
    'batch_size': 256,
    'epochs': 100,
    'lr': 1e-3,
    'optimizer': 'Adam',  # 支持 'Adam', 'SGD'
    
    # 数据路径 (建议使用绝对路径或相对于脚本的路径)
    'data_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SeversonBattery.mat'),
    
    # 输出目录
    'output_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_results'),
    
    # 设备
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 模型通用参数
    'seq_len': 1,
    'perc_val': 0.2,
    
    # Baseline / DeepHPM 参数
    'num_layers': 1,
    'num_neurons': 128,
    'hidden_layers': [128],
    'activation': 'Tanh',
    
    # BiLSTM 特有参数
    'lstm_layers': 1,
    'lstm_hidden_dim': 128,
    
    # DeepHPM 特有参数 (物理约束相关)
    # 示例: 'torch.cat((t, s), dim=2)'
    'inputs_dynamical': 'torch.cat((t, s), dim=2)',
    'inputs_dim_dynamical': 2, # 根据 inputs_dynamical 的维度设定
    
    # 学习率调度
    'step_size': 50,
    'gamma': 0.1,
}
# ==============================================

# ================= Monkey Patching =================
class PatchedNeuralNet(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, activation='Tanh'):
        super(PatchedNeuralNet, self).__init__()

        activation = USER_CONFIG.get('activation', activation)
        if activation not in ('Tanh', 'Sin'):
            raise ValueError(f"Unsupported activation: {activation}")

        self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim

        self.layers = []

        self.layers.append(nn.Linear(in_features=inputs_dim, out_features=layers[0]))
        nn.init.xavier_normal_(self.layers[-1].weight)

        if activation == 'Tanh':
            self.layers.append(nn.Tanh())
        elif activation == 'Sin':
            self.layers.append(func.Sin())
        self.layers.append(nn.Dropout(p=0.2))

        for l in range(len(layers) - 1):
            self.layers.append(nn.Linear(in_features=layers[l], out_features=layers[l + 1]))
            nn.init.xavier_normal_(self.layers[-1].weight)

            if activation == 'Tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'Sin':
                self.layers.append(func.Sin())
            self.layers.append(nn.Dropout(p=0.2))

        if len(layers) > 0:
            self.layers.append(nn.Linear(in_features=layers[-1], out_features=outputs_dim))
        else:
            self.layers.append(nn.Linear(in_features=inputs_dim, out_features=outputs_dim))
        nn.init.xavier_normal_(self.layers[-1].weight)

        self.NN = nn.Sequential(*self.layers)

    def forward(self, x):
        self.x = x
        self.x.requires_grad_(True)
        self.x_2D = self.x.contiguous().view((-1, self.inputs_dim))
        NN_out_2D = self.NN(self.x_2D)
        self.u_pred = NN_out_2D.contiguous().view((-1, self.seq_len, self.outputs_dim))

        return self.u_pred

func.Neural_Net = PatchedNeuralNet
# ===============================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ---------- BiLSTM 模型定义 (来自 SOH_CaseA_BiLSTM.py) ----------
class BiLSTMModel(nn.Module):
    """
    双向LSTM模型用于电池健康状态（SoH）预测。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        # 输入维度是hidden_dim*2，因为是双向LSTM
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # out: (batch_size, seq_len, hidden_dim*2)
        out, _ = self.lstm(x)
        # 取最后一个时间步
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(1)

def run_training():
    cfg = USER_CONFIG
    print(f"Starting local training with config: {cfg}")
    
    ensure_dir(cfg['output_dir'])
    device = torch.device(cfg['device'])
    
    # 1. 加载数据
    if not os.path.exists(cfg['data_path']):
        raise FileNotFoundError(f"Data file not found at {cfg['data_path']}")
        
    print(f"Loading data from {cfg['data_path']}...")
    data = func.SeversonBattery(cfg['data_path'], seq_len=cfg['seq_len'])
    
    # 2. 数据划分
    # 使用与案例代码相同的划分逻辑
    inputs_dict, targets_dict = func.create_chosen_cells(
        data,
        idx_cells_train=[91, 100, 124] if cfg['model_type'] == 'BiLSTM' else [91, 100], # BiLSTM 案例中用了更多训练数据，这里做个示例调整，或者统一
        idx_cells_test=[124] if cfg['model_type'] != 'BiLSTM' else [100], # BiLSTM 案例测试集不同
        perc_val=cfg['perc_val']
    )
    
    # 统一化：为了简单，我们使用更通用的划分，或者遵循 Baseline 的默认
    # 这里为了演示，我们强制使用 Baseline 的划分，除非用户改代码
    # 如果想完全复刻各脚本行为，需要写if/else。这里稍微做一点统一以适应通用脚本。
    # 实际上，BiLSTM 脚本里是 idx_cells_train=[91, 124], idx_cells_test=[100]
    # Baseline 脚本里是 idx_cells_train=[91, 100], idx_cells_test=[124]
    # 我们根据模型类型自动切换，以尽量还原
    if cfg['model_type'] == 'BiLSTM':
        train_cells = [91, 124]
        test_cells = [100]
    else:
        train_cells = [91, 100]
        test_cells = [124]
        
    print(f"Splitting data: Train cells {train_cells}, Test cells {test_cells}")
    inputs_dict, targets_dict = func.create_chosen_cells(
        data,
        idx_cells_train=train_cells,
        idx_cells_test=test_cells,
        perc_val=cfg['perc_val']
    )

    inputs_train = inputs_dict['train'].to(device)
    inputs_val = inputs_dict['val'].to(device)
    inputs_test = inputs_dict['test'].to(device)
    
    # 目标通常取第一列 (PCL/Capacity Loss)
    targets_train = targets_dict['train'][:, :, 0:1].to(device)
    targets_val = targets_dict['val'][:, :, 0:1].to(device)
    targets_test = targets_dict['test'][:, :, 0:1].to(device)
    
    inputs_dim = inputs_train.shape[2]
    outputs_dim = 1
    
    # 3. 数据标准化 (除了 BiLSTM 似乎没显式做 func.standardize_tensor 的 fit，而是直接输入？)
    # 检查代码：
    # Baseline: 做了 standardize_tensor(..., mode='fit') 并传入 DataDrivenNN
    # DeepHPM: 做了 standardize_tensor(..., mode='fit') 并传入 DeepHPMNN
    # BiLSTM: 没做显式标准化传入模型？BiLSTMModel 内部没有 scaler。
    # 但是 BiLSTM 脚本里 inputs_train 也是直接用的。
    # 仔细看 SOH_CaseA_BiLSTM.py，确实没有 standardize_tensor。
    # 这意味着 BiLSTM 可能直接学习原始特征，或者数据在加载时已经有一定处理。
    # 为了保持一致性，我们对 Baseline/DeepHPM 做标准化计算。
    
    mean_inputs, std_inputs = None, None
    mean_targets, std_targets = None, None
    
    if cfg['model_type'] in ['Baseline', 'DeepHPM']:
        _, mean_inputs, std_inputs = func.standardize_tensor(inputs_train, mode='fit')
        _, mean_targets, std_targets = func.standardize_tensor(targets_train, mode='fit')

    # 4. 构建模型
    print(f"Building {cfg['model_type']} model...")
    if cfg['model_type'] == 'Baseline':
        layers = list(cfg.get('hidden_layers') or (cfg['num_layers'] * [cfg['num_neurons']]))
        model = func.DataDrivenNN(
            seq_len=cfg['seq_len'],
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=layers,
            scaler_inputs=(mean_inputs, std_inputs),
            scaler_targets=(mean_targets, std_targets),
        ).to(device)
        criterion = func.My_loss(mode='Baseline')
        
    elif cfg['model_type'] == 'DeepHPM':
        layers = list(cfg.get('hidden_layers') or (cfg['num_layers'] * [cfg['num_neurons']]))
        model = func.DeepHPMNN(
            seq_len=cfg['seq_len'],
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=layers,
            scaler_inputs=(mean_inputs, std_inputs),
            scaler_targets=(mean_targets, std_targets),
            inputs_dynamical=cfg['inputs_dynamical'],
            inputs_dim_dynamical=cfg['inputs_dim_dynamical']
        ).to(device)
        criterion = func.My_loss(mode='Sum')
        
    elif cfg['model_type'] == 'BiLSTM':
        # BiLSTM 脚本中禁用了 cudnn
        if device.type == 'cuda':
            torch.backends.cudnn.enabled = False
            
        model = BiLSTMModel(
            input_dim=inputs_dim,
            hidden_dim=cfg['lstm_hidden_dim'],
            output_dim=outputs_dim,
            num_layers=cfg['lstm_layers']
        ).to(device)
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown model type: {cfg['model_type']}")

    # 5. 优化器和调度器
    optimizer_cls = getattr(optim, cfg['optimizer'])
    optimizer = optimizer_cls(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
    
    # 6. 训练循环
    print("Starting training...")
    
    # 准备 DataLoader
    train_set = TensorDataset(inputs_train, targets_train) # 简单封装，具体 func.train 里可能还有逻辑
    # 注意：func.train 接受 train_loader
    # BiLSTM 是自己写的循环，Baseline/DeepHPM 用 func.train
    
    train_loader = DataLoader(
        train_set,
        batch_size=cfg['batch_size'],
        shuffle=True,
        drop_last=True
    )
    
    history = {'train_loss': [], 'val_loss': []}
    
    if cfg['model_type'] == 'BiLSTM':
        # BiLSTM 自定义训练循环
        for epoch in range(cfg['epochs']):
            model.train()
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            
            # 简单验证
            model.eval()
            with torch.no_grad():
                val_pred = model(inputs_val)
                val_loss = criterion(val_pred, targets_val).item()
                
            print(f"Epoch [{epoch+1}/{cfg['epochs']}] Train Loss: {epoch_loss/len(train_loader):.6f} Val Loss: {val_loss:.6f}")
            history['train_loss'].append(epoch_loss/len(train_loader))
            history['val_loss'].append(val_loss)
            
    else:
        # Baseline / DeepHPM 使用 func.train
        # func.train 需要一些额外参数
        log_sigma_u = torch.zeros((), device=device)
        log_sigma_f = torch.zeros((), device=device)
        log_sigma_f_t = torch.zeros((), device=device)
        
        # func.train 返回 (model, results_epoch)
        # 注意：func.train 内部包含了 epoch 循环
        model, results_epoch = func.train(
            num_epoch=cfg['epochs'],
            batch_size=cfg['batch_size'],
            train_loader=train_loader,
            num_slices_train=inputs_train.shape[0],
            inputs_val=inputs_val,
            targets_val=targets_val,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            log_sigma_u=log_sigma_u,
            log_sigma_f=log_sigma_f,
            log_sigma_f_t=log_sigma_f_t
        )
        # func.train 已经打印了进度
        
    # 7. 评估与保存
    print("Evaluating on test set...")
    model.eval()
    
    if cfg['model_type'] == 'DeepHPM':
        # DeepHPM forward 返回 (U, F, F_t)
        U_pred, _, _ = model(inputs_test)
    elif cfg['model_type'] == 'Baseline':
         # Baseline forward 返回 (U, F, F_t) (虽然 F/F_t 可能是 dummy)
        U_pred, _, _ = model(inputs_test)
    else: # BiLSTM
        U_pred = model(inputs_test)
        
    # 后处理：转换回 SoH (假设 targets 是 Capacity Loss, SoH = 1 - Loss)
    # 注意：Baseline/DeepHPM 代码里做了 1. - output
    # BiLSTM 代码里也做了 1. - output
    # 我们这里统一做
    
    U_pred_soh = 1.0 - U_pred
    targets_test_soh = 1.0 - targets_test
    
    # 计算 Metrics
    mse = torch.mean((U_pred_soh - targets_test_soh) ** 2).item()
    rmspe = torch.sqrt(torch.mean(((U_pred_soh - targets_test_soh) / targets_test_soh) ** 2)).item()
    
    print(f"Final Test MSE: {mse:.6f}")
    print(f"Final Test RMSPE: {rmspe:.6f}")
    
    # 保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(cfg['output_dir'], f"{cfg['model_type']}_{timestamp}.pth")
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'metrics': {'mse': mse, 'rmspe': rmspe},
        'predictions': U_pred_soh.detach().cpu().numpy(),
        'targets': targets_test_soh.detach().cpu().numpy()
    }
    
    # 对于 Baseline/DeepHPM，保存 scaler 以便未来推理
    if mean_inputs is not None:
        save_dict['scaler_inputs'] = (mean_inputs, std_inputs)
        save_dict['scaler_targets'] = (mean_targets, std_targets)
        
    torch.save(save_dict, save_path)
    print(f"Model and results saved to {save_path}")

if __name__ == '__main__':
    run_training()
