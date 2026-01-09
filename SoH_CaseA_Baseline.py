# 导入必要的库
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import functions as func

# 检测是否有可用的GPU设备，如果有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预先配置的超参数设置
settings = torch.load('./Settings/settings_SoH_CaseA.pth')
print(settings)
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
# 从设置中获取网络层数列表
num_layers = settings['num_layers']
# 从设置中获取每层神经元数列表
num_neurons = settings['num_neurons']

# 加载Severson电池数据集
# addr = './SeversonBattery.mat'
addr='./2017-05-12_batchdata_updated_struct_errorcorrect.mat'
data = func.SeversonBattery(addr, seq_len=seq_len)

# 在超参数搜索之前创建一次数据分割
# 选择训练单元[91, 100]和测试单元[124]，验证集占训练集的20%
inputs_dict, targets_dict = func.create_chosen_cells(
    data,
    idx_cells_train=[91, 100],
    idx_cells_test=[124],
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

# 初始化用于存储评估指标的字典
metric_mean = dict()
metric_std = dict()
# 初始化训练集评估指标的平均值矩阵（行：网络层数，列：神经元数）
metric_mean['train'] = np.zeros((len(num_layers), len(num_neurons)))
# 初始化验证集评估指标的平均值矩阵
metric_mean['val'] = np.zeros((len(num_layers), len(num_neurons)))
# 初始化测试集评估指标的平均值矩阵
metric_mean['test'] = np.zeros((len(num_layers), len(num_neurons)))
# 初始化训练集评估指标的标准差矩阵
metric_std['train'] = np.zeros((len(num_layers), len(num_neurons)))
# 初始化验证集评估指标的标准差矩阵
metric_std['val'] = np.zeros((len(num_layers), len(num_neurons)))
# 初始化测试集评估指标的标准差矩阵
metric_std['test'] = np.zeros((len(num_layers), len(num_neurons)))
# 遍历所有网络层数配置
for l, num_l in enumerate(num_layers):
    # 遍历所有神经元数配置
    for n, num_n in enumerate(num_neurons):
        # 创建网络结构：num_l层，每层num_n个神经元
        layers = num_l * [num_n]
        # 设置NumPy随机种子以确保可重现性
        np.random.seed(1234)
        # 设置PyTorch随机种子
        torch.manual_seed(1234)
        # 初始化用于存储每一轮结果的字典
        metric_rounds = dict()
        metric_rounds['train'] = np.zeros(num_rounds)
        metric_rounds['val'] = np.zeros(num_rounds)
        metric_rounds['test'] = np.zeros(num_rounds)
        # 进行多轮实验
        for round in range(num_rounds):
            # 获取输入的特征维度
            inputs_dim = inputs_train.shape[2]
            # 设置输出维度为1（预测一个值：容量衰减）
            outputs_dim = 1

            # 计算训练输入的均值和标准差
            _, mean_inputs_train, std_inputs_train = func.standardize_tensor(inputs_train, mode='fit')
            # 计算训练目标的均值和标准差
            _, mean_targets_train, std_targets_train = func.standardize_tensor(targets_train, mode='fit')

            # 创建训练数据集
            train_set = func.TensorDataset(inputs_train, targets_train)  # J_train is a placeholder
            # 创建训练数据加载器
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )

            # 创建数据驱动神经网络模型
            model = func.DataDrivenNN(
                seq_len=seq_len,
                inputs_dim=inputs_dim,
                outputs_dim=outputs_dim,
                layers=layers,
                scaler_inputs=(mean_inputs_train, std_inputs_train),
                scaler_targets=(mean_targets_train, std_targets_train),
            ).to(device)

            # 初始化不确定性参数（基线模式下不使用）
            log_sigma_u = torch.zeros(())
            log_sigma_f = torch.zeros(())
            log_sigma_f_t = torch.zeros(())

            # 创建损失函数（基线模式：仅使用数据拟合损失）
            criterion = func.My_loss(mode='Baseline')

            # 获取模型参数
            params = ([p for p in model.parameters()])
            # 创建Adam优化器
            optimizer = optim.Adam(params, lr=settings['lr'])
            # 创建学习率调度器（每step_size个epoch后乘以gamma）
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=settings['step_size'], gamma=settings['gamma'])

            # 训练模型
            model, results_epoch = func.train(
                num_epoch=num_epoch,
                batch_size=batch_size,
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

            # 设置模型为评估模式
            model.eval()

            # 在训练集上进行评估
            U_pred_train, F_pred_train, _ = model(inputs=inputs_train)
            # 将容量损失转换为健康状态（SoH）
            U_pred_train_soh = 1. - U_pred_train
            targets_train_soh = 1. - targets_train
            # 计算训练集RMSPE（相对均方根百分比误差）
            RMSPE_train = torch.sqrt(torch.mean(((U_pred_train_soh - targets_train_soh) / targets_train_soh) ** 2))
            # 计算训练集MSE（均方误差）
            MSE_train = torch.mean((U_pred_train_soh - targets_train_soh) ** 2)
            # 计算总平方和的残差
            SS_res = torch.sum((U_pred_train_soh - targets_train_soh) ** 2)
            # 计算总平方和
            SS_tot = torch.sum((targets_train_soh - torch.mean(targets_train_soh)) ** 2)
            # 计算R²分数
            R2_train = 1 - (SS_res / SS_tot)

            # 在验证集上进行评估
            U_pred_val, F_pred_val, _ = model(inputs=inputs_val)
            # 将容量损失转换为健康状态（SoH）
            U_pred_val_soh = 1. - U_pred_val
            targets_val_soh = 1. - targets_val
            # 计算验证集RMSPE
            RMSPE_val = torch.sqrt(torch.mean(((U_pred_val_soh - targets_val_soh) / targets_val_soh) ** 2))
            # 计算验证集MSE
            MSE_val = torch.mean((U_pred_val_soh - targets_val_soh) ** 2)
            # 计算验证集残差平方和
            SS_res_val = torch.sum((U_pred_val_soh - targets_val_soh) ** 2)
            # 计算验证集总平方和
            SS_tot_val = torch.sum((targets_val_soh - torch.mean(targets_val_soh)) ** 2)
            # 计算验证集R²分数
            R2_val = 1 - (SS_res_val / SS_tot_val)

            # 在测试集上进行评估
            U_pred_test, F_pred_test, _ = model(inputs=inputs_test)
            # 将容量损失转换为健康状态（SoH）
            U_pred_test_soh = 1. - U_pred_test
            targets_test_soh = 1. - targets_test
            # 计算测试集RMSPE
            RMSPE_test = torch.sqrt(torch.mean(((U_pred_test_soh - targets_test_soh) / targets_test_soh) ** 2))
            # 计算测试集MSE
            MSE_test = torch.mean((U_pred_test_soh - targets_test_soh) ** 2)
            # 计算测试集残差平方和
            SS_res_test = torch.sum((U_pred_test_soh - targets_test_soh) ** 2)
            # 计算测试集总平方和
            SS_tot_test = torch.sum((targets_test_soh - torch.mean(targets_test_soh)) ** 2)
            # 计算测试集R²分数
            R2_test = 1 - (SS_res_test / SS_tot_test)

            # 将当前轮次的RMSPE指标保存
            metric_rounds['train'][round] = RMSPE_train.detach().cpu().numpy()
            metric_rounds['val'][round] = RMSPE_val.detach().cpu().numpy()
            metric_rounds['test'][round] = RMSPE_test.detach().cpu().numpy()

        # 计算该超参数配置下的平均指标
        metric_mean['train'][l, n] = np.mean(metric_rounds['train'])
        metric_mean['val'][l, n] = np.mean(metric_rounds['val'])
        metric_mean['test'][l, n] = np.mean(metric_rounds['test'])
        # 计算该超参数配置下的标准差
        metric_std['train'][l, n] = np.std(metric_rounds['train'])
        metric_std['val'][l, n] = np.std(metric_rounds['val'])
        metric_std['test'][l, n] = np.std(metric_rounds['test'])

# 在最后一个训练模型上进行最终评估
# 注：由于num_rounds >= 1且num_layers/num_neurons至少有一个元素，所以模型一定被定义
model.eval()  # type: ignore
# 在测试集上进行前向传播
U_pred_test_final, F_pred_test_final, _ = model(inputs=inputs_test)  # type: ignore
# 将预测的容量损失转换为健康状态（SoH）
U_pred_test_final_soh = 1. - U_pred_test_final
# 将目标容量损失转换为目标健康状态（SoH）
targets_test_final_soh = 1. - targets_test

# 计算最终的评估指标
# 计算最终RMSPE
RMSPE_test_final = torch.sqrt(torch.mean(((U_pred_test_final_soh - targets_test_final_soh) / targets_test_final_soh) ** 2))
# 计算最终MSE
MSE_test_final = torch.mean((U_pred_test_final_soh - targets_test_final_soh) ** 2)
# 计算最终残差平方和
SS_res_final = torch.sum((U_pred_test_final_soh - targets_test_final_soh) ** 2)
# 计算最终总平方和
SS_tot_final = torch.sum((targets_test_final_soh - torch.mean(targets_test_final_soh)) ** 2)
# 计算最终R²分数
R2_test_final = 1 - (SS_res_final / SS_tot_final)

# 保存实验结果
results = dict()
# 保存真实的SoH值
results['U_true'] = targets_test_final_soh.detach().cpu().numpy().squeeze()
# 保存预测的SoH值
results['U_pred'] = U_pred_test_final_soh.detach().cpu().numpy().squeeze()
# 保存SoH关于时间的导数
results['U_t_pred'] = model.U_t.detach().cpu().numpy().squeeze()  # type: ignore
# 保存循环次数
results['Cycles'] = inputs_test[:, :, -1:].detach().cpu().numpy().squeeze()
# 保存训练轮数
results['Epochs'] = np.arange(0, num_epoch)
# 保存所有超参数配置的平均指标
results['metric_mean'] = metric_mean
# 保存所有超参数配置的标准差
results['metric_std'] = metric_std
# 将结果保存到文件
torch.save(results, './results/SoH_CaseA_Baseline.pth')

# 打印最终的测试指标
print(f"Test RMSPE: {RMSPE_test_final.item()}, Test MSE: {MSE_test_final.item()}, R^2: {R2_test_final.item()}")
# 打印所有超参数配置下的平均RMSPE
print(f"Mean Test RMSPE across hyperparameters: {metric_mean['test']}")
# 打印所有超参数配置下的RMSPE标准差
print(f"Std Test RMSPE across hyperparameters: {metric_std['test']}")
