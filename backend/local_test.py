# -*- coding: utf-8 -*-
"""
本地算法测试脚本
用于加载训练好的模型 (.pth) 并对指定电池进行 RUL 预测和绘图评估。
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import ast

# 引入项目根目录以导入 functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions as func

# ================= 测试配置 =================
TEST_CONFIG = {
    # 待测试的电池索引列表 (例如 [15] 代表第16块电池)
    'test_cells': [32],
    
    # 模型文件路径 (请修改为实际路径)
    'model_path': r'g:\学习\大四课设\电池\algorithm_training_platform\results\DeepHPM_20260113104230.pth',
    
    # 数据集路径
    'data_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SeversonBattery.mat'),
    
    # 绘图采样步长
    'step': 1,
    
    # 运行设备
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
# ===========================================

# ================= 模型定义与补丁 =================
# 必须与训练时的模型定义保持一致，包括 Monkey Patch

class PatchedNeuralNet(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, activation='Tanh'):
        super(PatchedNeuralNet, self).__init__()
        
        self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim
        self.layers = []
        
        # 激活函数选择
        if activation == 'Sin':
            act_func = func.Sin()
        else:
            act_func = nn.Tanh()

        # 构建层
        # 输入层
        self.layers.append(nn.Linear(in_features=inputs_dim, out_features=layers[0]))
        self.layers.append(act_func)
        self.layers.append(nn.Dropout(p=0.2))

        # 隐藏层
        for l in range(len(layers) - 1):
            self.layers.append(nn.Linear(in_features=layers[l], out_features=layers[l + 1]))
            self.layers.append(act_func)
            self.layers.append(nn.Dropout(p=0.2))

        # 输出层
        if len(layers) > 0:
            self.layers.append(nn.Linear(in_features=layers[-1], out_features=outputs_dim))
        else:
            self.layers.append(nn.Linear(in_features=inputs_dim, out_features=outputs_dim))

        self.NN = nn.Sequential(*self.layers)

    def forward(self, x):
        # 调整输入形状以适应全连接层
        x_2D = x.contiguous().view((-1, self.inputs_dim))
        out_2D = self.NN(x_2D)
        # 调整回序列形状
        return out_2D.contiguous().view((-1, self.seq_len, self.outputs_dim))

# 应用补丁替换原始 Neural_Net
func.Neural_Net = PatchedNeuralNet


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(1)

# ================================================

def load_config_safely(checkpoint):
    """安全解析 checkpoint 中的配置"""
    # 优先使用字典格式的 train_config，其次尝试解析字符串 config
    train_cfg = checkpoint.get('train_config') or checkpoint.get('config') or {}
    
    if isinstance(train_cfg, str):
        print("检测到配置为字符串格式，正在解析...")
        try:
            # 尝试使用 ast.literal_eval (最安全)
            train_cfg = ast.literal_eval(train_cfg)
        except Exception:
            try:
                # 回退到 eval，并提供 TrainConfig 的占位解析
                class TrainConfig:
                    def __init__(self, **kwargs):
                        self.__dict__.update(kwargs)
                context = {'TrainConfig': TrainConfig}
                obj = eval(train_cfg, context)
                train_cfg = getattr(obj, '__dict__', {})
            except Exception as e:
                print(f"配置解析失败: {e}。将使用默认参数。")
                train_cfg = {}
    
    return train_cfg if isinstance(train_cfg, dict) else {}

def main():
    cfg = TEST_CONFIG
    device = torch.device(cfg['device'])
    
    # 1. 加载 Checkpoint
    if not os.path.exists(cfg['model_path']):
        print(f"错误: 模型文件不存在: {cfg['model_path']}")
        return

    print(f"正在加载模型: {cfg['model_path']} ...")
    # weights_only=False 是为了支持加载包含 numpy/config 对象的旧版 checkpoint
    checkpoint = torch.load(cfg['model_path'], map_location=device, weights_only=False)
    
    # 2. 解析训练配置
    train_cfg = load_config_safely(checkpoint)
    model_type = train_cfg.get('model_type', 'Baseline')
    print(f"识别模型类型: {model_type}")

    # 3. 加载数据
    print(f"正在加载数据集: {cfg['data_path']} ...")
    seq_len = train_cfg.get('seq_len', 1)
    data = func.SeversonBattery(cfg['data_path'], seq_len=seq_len)
    
    # 4. 遍历测试电池
    plt.figure(figsize=(14, 10 * len(cfg['test_cells'])))
    
    model = None # 确保模型只构建一次
    
    for i, cell_id in enumerate(cfg['test_cells']):
        if not (0 <= cell_id < data.num_cells):
            print(f"警告: 电池索引 {cell_id} 超出范围 (0-{data.num_cells-1})，跳过。")
            continue
            
        print(f"\n=== 正在处理电池索引: {cell_id} ===")
        
        # 获取原始数据 (不经过 train/test 划分)
        raw_input = data.inputs_units[cell_id]
        raw_target = data.targets_units[cell_id]
        
        # 创建切片 (用于模型输入)
        inputs_list, targets_list, _ = func.create_slices(
            data_units=[raw_input],
            RUL_units=[raw_target],
            seq_len_slices=seq_len,
            steps_slices=cfg['step']
        )
        
        # 转换为 Tensor
        inputs = torch.from_numpy(inputs_list[0]).type(torch.float32).to(device)
        targets = torch.from_numpy(targets_list[0]).type(torch.float32).to(device)
        # 仅使用 PCL 目标通道 (第 0 通道)，与模型输出维度对齐
        targets = targets[:, :, 0:1]
        
        # 确保输入维度正确 (batch, seq_len, input_dim)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
            
        inputs_dim = inputs.shape[2]
        outputs_dim = 1
        
        # 5. 构建模型 (如果尚未构建)
        if model is None:
            print("正在构建并加载模型权重...")
            scaler_inputs = checkpoint.get('scaler_inputs')
            scaler_targets = checkpoint.get('scaler_targets')
            
            # 提取网络参数
            num_neurons = train_cfg.get('num_neurons', 128)
            num_layers = train_cfg.get('num_layers', 1)
            layers_cfg = train_cfg.get('layers')
            hidden_layers = layers_cfg if layers_cfg else train_cfg.get('hidden_layers')
            if not hidden_layers:
                hidden_layers = [num_neurons] * num_layers
            
            if model_type == 'Baseline':
                model = func.DataDrivenNN(
                    seq_len=seq_len,
                    inputs_dim=inputs_dim,
                    outputs_dim=outputs_dim,
                    layers=hidden_layers,
                    scaler_inputs=scaler_inputs,
                    scaler_targets=scaler_targets
                ).to(device)
                
            elif model_type == 'DeepHPM':
                # 处理 DeepHPM 特有参数
                inputs_dynamical = train_cfg.get('inputs_dynamical', 'torch.cat((t, s), dim=2)')
                inputs_dim_dynamical = train_cfg.get('inputs_dim_dynamical', 2)
                
                model = func.DeepHPMNN(
                    seq_len=seq_len,
                    inputs_dim=inputs_dim,
                    outputs_dim=outputs_dim,
                    layers=hidden_layers,
                    scaler_inputs=scaler_inputs,
                    scaler_targets=scaler_targets,
                    inputs_dynamical=inputs_dynamical,
                    inputs_dim_dynamical=inputs_dim_dynamical
                ).to(device)
                
            elif model_type == 'BiLSTM':
                lstm_hidden = train_cfg.get('lstm_hidden_dim', train_cfg.get('hidden_dim', 128))
                lstm_layers = train_cfg.get('lstm_layers', train_cfg.get('num_layers', 1))
                
                model = BiLSTMModel(
                    input_dim=inputs_dim,
                    hidden_dim=lstm_hidden,
                    output_dim=outputs_dim,
                    num_layers=lstm_layers
                ).to(device)
            
            # 加载权重
            sd = checkpoint['model_state_dict']
            if any(k.startswith('model.') for k in sd.keys()):
                sd = {k.replace('model.', ''): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            model.eval()
            print("模型加载完成。")

        # 6. 推理预测
        if model_type == 'BiLSTM':
            with torch.no_grad():
                U_pred = model(inputs)
        else:
            # Baseline / DeepHPM 需要 autograd 计算 U_t，不能禁用梯度
            U_pred, _, _ = model(inputs)
        
        # 7. 数据后处理为 PCL 与 RUL
        U_pred_np = U_pred.detach().cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        
        pcl_pred = U_pred_np
        pcl_true = targets_np
        
        # 8. 计算 EOL 与 RUL（以 PCL 阈值定义）
        pcl_threshold = 0.2
        
        def calculate_eol_idx_pcl(pcl_series, thresh):
            indices = np.where(pcl_series > thresh)[0]
            if len(indices) > 0:
                return indices[0]
            return len(pcl_series)
        
        true_eol = calculate_eol_idx_pcl(pcl_true, pcl_threshold)
        pred_eol = calculate_eol_idx_pcl(pcl_pred, pcl_threshold)
        error = pred_eol - true_eol
        
        print(f"预测结果: 真实EOL={true_eol}, 预测EOL={pred_eol}, 误差={error} cycles")
        
        # 9. 绘制 PCL 与 RUL 曲线（每个电池两个子图）
        cycles = np.arange(len(pcl_true)) + 1
        rows = len(cfg['test_cells']) * 2
        base = i * 2
        
        ax1 = plt.subplot(rows, 1, base + 1)
        ax1.plot(cycles, pcl_true, 'k-', label='Actual PCL', linewidth=1.5)
        ax1.plot(cycles, pcl_pred, 'r--', label='Predicted PCL', linewidth=1.5)
        ax1.axhline(y=pcl_threshold, color='g', linestyle=':', label='EOL PCL Threshold (0.2)')
        ax1.text(0.02, 0.05, f"EOL Error: {error} cycles\n(True: {true_eol}, Pred: {pred_eol})",
                 transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        ax1.set_title(f'Battery Cell {cell_id} - PCL Prediction ({model_type})')
        ax1.set_ylabel('PCL')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        eol_true_cycle = true_eol + 1
        eol_pred_cycle = pred_eol + 1
        rul_true = np.maximum(eol_true_cycle - cycles, 0)
        rul_pred = np.maximum(eol_pred_cycle - cycles, 0)
        
        ax2 = plt.subplot(rows, 1, base + 2)
        ax2.plot(cycles, rul_true, 'k-', label='Actual RUL', linewidth=1.5)
        ax2.plot(cycles, rul_pred, 'r--', label='Predicted RUL', linewidth=1.5)
        ax2.set_title(f'Battery Cell {cell_id} - RUL Prediction ({model_type})')
        ax2.set_ylabel('RUL (cycles)')
        ax2.set_xlabel('Cycle')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
