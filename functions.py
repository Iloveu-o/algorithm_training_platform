# -*- coding: utf-8 -*-
"""  
电池健康状态(SoH)估算函数模块

此模块包含用于实现电池健康状态估算神经网络模型的类和函数。
包括数据处理、模型架构、训练工具以及各种方法的损失函数，
包括基线方法、物理信息神经网络(PINNs)和深度混合物理-经验模型。
"""

import numpy as np
import scipy.io
import h5py
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset






class SeversonBattery:
    """
    Severson电池数据集的数据加载器和预处理器。
    
    此类从.mat文件加载电池循环数据，并通过组织输入和目标、创建数据切片、
    以及处理训练/验证/测试分割，为神经网络训练做准备。
    
    属性:
        data: 从.mat文件加载的原始数据
        seq_len: 输入数据的序列长度
        steps_slices: 创建数据切片的步长
        features: 电池特征（电压、电流、温度等）
        RUL: 剩余使用寿命值
        PCL: 容量衰减百分比值
        cycles: 循环次数
        idx_train_units: 训练单元的索引
        idx_val_units: 验证单元的索引
        idx_test_units: 测试单元的索引
        inputs: 组合的特征和循环次数
        targets: 组合的PCL和RUL值
        inputs_dim: 输入维度数量
        targets_dim: 目标维度数量
        num_cycles_all: 每个电池单元的循环次数
        num_cells: 电池单元总数
        inputs_units: 每个单元的处理后的输入数据
        targets_units: 每个单元的处理后的目标数据
        inputs_train_units: 每个单元的训练输入
        targets_train_units: 每个单元的训练目标
        inputs_val_units: 每个单元的验证输入
        targets_val_units: 每个单元的验证目标
        inputs_test_units: 每个单元的测试输入
        targets_test_units: 每个单元的测试目标
        inputs_train_slices: 训练输入切片
        targets_train_slices: 训练目标切片
        inputs_val_slices: 验证输入切片
        targets_val_slices: 验证目标切片
        inputs_test_slices: 测试输入切片
        targets_test_slices: 测试目标切片
        num_slices_lib_train: 训练数据中的切片数量
        num_slices_lib_val: 验证数据中的切片数量
        num_slices_lib_test: 测试数据中的切片数量
        num_slices_train: 训练切片总数
        inputs_train_ndarray: 训练输入的numpy数组
        targets_train_ndarray: 训练目标的numpy数组
        num_slices_val: 验证切片总数
        inputs_val_ndarray: 验证输入的numpy数组
        targets_val_ndarray: 验证目标的numpy数组
        num_slices_test: 测试切片总数
        inputs_test_ndarray: 测试输入的numpy数组
        targets_test_ndarray: 测试目标的numpy数组
        inputs_train_tensor: 训练输入的torch张量
        inputs_val_tensor: 验证输入的torch张量
        inputs_test_tensor: 测试输入的torch张量
        targets_train_tensor: 训练目标的torch张量
        targets_val_tensor: 验证目标的torch张量
        targets_test_tensor: 测试目标的torch张量
    """
    def __init__(self, data_addr, seq_len):
        # 尝试加载.mat格式的电池数据文件
        # 首先尝试使用scipy.io.loadmat（适用于MATLAB v7及更早版本）
        try:
            self.data = scipy.io.loadmat(data_addr)
        except NotImplementedError:
            # 如果是MATLAB v7.3格式，则使用h5py读取
            print("检测到MATLAB v7.3格式文件，使用h5py读取...")
            self.data = self._load_h5_file(data_addr)
        
        # 设置序列长度
        self.seq_len = seq_len
        # 设置切片步长
        self.steps_slices = 1

        # 提取电池特征数据（如电压、电流、温度等）
        self.features = self.data['Features_mov_Flt']
        # 提取剩余使用寿命(RUL)数据
        self.RUL = self.data['RUL_Flt']
        # 提取容量衰减百分比(PCL)数据
        self.PCL = self.data['PCL_Flt']
        # 提取循环次数数据
        self.cycles = self.data['Cycles_Flt']


        # 提取剩余使用寿命(RUL)数据
        self.RUL = self.data['RUL_Flt']
        # 提取容量衰减百分比(PCL)数据
        self.PCL = self.data['PCL_Flt']
        # 提取循环次数数据
        self.cycles = self.data['Cycles_Flt']

        # 获取训练单元的索引（减1是因为MATLAB索引从1开始）
        self.idx_train_units = self.data['train_ind'].flatten() - 1
        # 获取验证单元的索引
        self.idx_val_units = self.data['test_ind'].flatten() - 1
        # 获取测试单元的索引
        self.idx_test_units = self.data['secondary_test_ind'].flatten() - 1

        # 合并特征和循环次数作为输入
        self.inputs = np.hstack((self.features, self.cycles.flatten()[:, None]))
        # 合并PCL和RUL作为目标值
        self.targets = np.hstack((self.PCL, self.RUL))
        # 计算输入维度
        self.inputs_dim = self.inputs.shape[1]
        # 计算目标维度
        self.targets_dim = self.targets.shape[1]

        # 获取每个电池单元的循环次数
        self.num_cycles_all = self.data['Num_Cycles_Flt'].flatten()
        # 计算电池单元总数
        self.num_cells = len(self.num_cycles_all)

        # 使用create_units函数组织数据
        self.inputs_units, self.targets_units = create_units(
            data=self.features,
            t=self.cycles,
            RUL=self.targets,
            num_units=self.num_cells,
            len_units=np.squeeze(self.num_cycles_all[:, None])
        )

        # 计算训练、验证、测试单元数量
        self.num_train_units = len(self.idx_train_units)
        self.num_val_units = len(self.idx_val_units)
        self.num_test_units = len(self.idx_test_units)

        # 初始化训练、验证、测试数据列表
        self.inputs_train_units = []
        self.targets_train_units = []
        self.inputs_val_units = []
        self.targets_val_units = []
        self.inputs_test_units = []
        self.targets_test_units = []

        # 为训练单元分配数据
        for i in range(self.num_train_units):
            self.inputs_train_units.append(self.inputs_units[self.idx_train_units[i]])
            self.targets_train_units.append(self.targets_units[self.idx_train_units[i]])

        # 为验证单元分配数据
        for i in range(self.num_val_units):
            self.inputs_val_units.append(self.inputs_units[self.idx_val_units[i]])
            self.targets_val_units.append(self.targets_units[self.idx_val_units[i]])

        # 为测试单元分配数据
        for i in range(self.num_test_units):
            self.inputs_test_units.append(self.inputs_units[self.idx_test_units[i]])
            self.targets_test_units.append(self.targets_units[self.idx_test_units[i]])

        # 为训练数据创建切片
        self.inputs_train_slices, self.targets_train_slices, self.num_slices_lib_train = create_slices(
            data_units=self.inputs_train_units,
            RUL_units=self.targets_train_units,
            seq_len_slices=self.seq_len,
            steps_slices=self.steps_slices
        )

        # 为验证数据创建切片
        self.inputs_val_slices, self.targets_val_slices, self.num_slices_lib_val = create_slices(
            data_units=self.inputs_val_units,
            RUL_units=self.targets_val_units,
            seq_len_slices=self.seq_len,
            steps_slices=self.steps_slices
        )

        # 为测试数据创建切片
        self.inputs_test_slices, self.targets_test_slices, self.num_slices_lib_test = create_slices(
            data_units=self.inputs_test_units,
            RUL_units=self.targets_test_units,
            seq_len_slices=self.seq_len,
            steps_slices=self.steps_slices
        )

        # 生成训练数据数组形式
        self.num_slices_train = np.sum(self.num_slices_lib_train)
        self.inputs_train_ndarray = np.zeros((self.num_slices_train, self.seq_len, self.inputs_dim))
        self.targets_train_ndarray = np.zeros((self.num_slices_train, self.seq_len, self.targets_dim))
        idx_start = 0
        for i in range(len(self.inputs_train_slices)):
            idx_end = idx_start + self.num_slices_lib_train[i]
            self.inputs_train_ndarray[idx_start:idx_end, :, :] = self.inputs_train_slices[i]
            self.targets_train_ndarray[idx_start:idx_end, :, :] = self.targets_train_slices[i]
            idx_start += self.num_slices_lib_train[i]

        # 生成验证数据数组形式
        self.num_slices_val = np.sum(self.num_slices_lib_val)
        self.inputs_val_ndarray = np.zeros((self.num_slices_val, self.seq_len, self.inputs_dim))
        self.targets_val_ndarray = np.zeros((self.num_slices_val, self.seq_len, self.targets_dim))
        idx_start = 0
        for i in range(len(self.inputs_val_slices)):
            idx_end = idx_start + self.num_slices_lib_val[i]
            self.inputs_val_ndarray[idx_start:idx_end, :, :] = self.inputs_val_slices[i]
            self.targets_val_ndarray[idx_start:idx_end, :, :] = self.targets_val_slices[i]
            idx_start += self.num_slices_lib_val[i]

        # 生成测试数据数组形式
        self.num_slices_test = np.sum(self.num_slices_lib_test)
        self.inputs_test_ndarray = np.zeros((self.num_slices_test, self.seq_len, self.inputs_dim))
        self.targets_test_ndarray = np.zeros((self.num_slices_test, self.seq_len, self.targets_dim))
        idx_start = 0
        for i in range(len(self.inputs_test_slices)):
            idx_end = idx_start + self.num_slices_lib_test[i]
            self.inputs_test_ndarray[idx_start:idx_end, :, :] = self.inputs_test_slices[i]
            self.targets_test_ndarray[idx_start:idx_end, :, :] = self.targets_test_slices[i]
            idx_start += self.num_slices_lib_test[i]

        # 将numpy数组转换为torch张量
        self.inputs_train_tensor = torch.from_numpy(self.inputs_train_ndarray).type(torch.float32)
        self.inputs_val_tensor = torch.from_numpy(self.inputs_val_ndarray).type(torch.float32)
        self.inputs_test_tensor = torch.from_numpy(self.inputs_test_ndarray).type(torch.float32)
        self.targets_train_tensor = torch.from_numpy(self.targets_train_ndarray).type(torch.float32)
        self.targets_val_tensor = torch.from_numpy(self.targets_val_ndarray).type(torch.float32)
        self.targets_test_tensor = torch.from_numpy(self.targets_test_ndarray).type(torch.float32)

    def _load_h5_file(self, data_addr):
        """
        使用h5py加载MATLAB v7.3格式的.mat文件。
        
        参数:
            data_addr: .mat文件的路径
            
        返回:
            data_dict: 包含加载数据的字典
        """
        data_dict = {}
        with h5py.File(data_addr, 'r') as f:
            # 首先打印文件结构，了解实际的键名和数据结构
            print("\n=== MATLAB v7.3文件结构 ===")
            print("\n文件中的所有键名：")
            
            def print_structure(name, obj):
                """递归打印HDF5文件结构"""
                if isinstance(obj, h5py.Dataset):
                    print(f"  数据集: {name}")
                    print(f"    - 形状: {obj.shape}")
                    print(f"    - 数据类型: {obj.dtype}")
                    # 如果是小数组，显示部分内容
                    if obj.size < 100:
                        print(f"    - 内容预览: {obj[()]}")
                elif isinstance(obj, h5py.Group):
                    print(f"  组: {name}")
            
            # 遍历文件中的所有对象
            f.visititems(print_structure)
            
            print("\n=== 尝试读取数据 ===")
            
            # 尝试读取所有顶层键
            for key in f.keys():
                print(f"\n正在处理键: {key}")
                try:
                    obj = f[key]
                    if isinstance(obj, h5py.Dataset):
                        data = np.array(obj)
                        print(f"  成功读取，形状: {data.shape}")
                        # 对于2D数据，需要转置（MATLAB列优先 -> NumPy行优先）
                        if data.ndim == 2:
                            data = data.T
                            print(f"  转置后形状: {data.shape}")
                        data_dict[key] = data
                    elif isinstance(obj, h5py.Group):
                        print(f"  这是一个组，包含: {list(obj.keys())}")
                        # 尝试读取组内的数据
                        for subkey in obj.keys():
                            try:
                                subdata = np.array(obj[subkey])
                                print(f"    子键 {subkey}: 形状 {subdata.shape}")
                                if subdata.ndim == 2:
                                    subdata = subdata.T
                                data_dict[f"{key}/{subkey}"] = subdata
                            except Exception as e:
                                print(f"    读取子键 {subkey} 失败: {e}")
                except Exception as e:
                    print(f"  读取失败: {e}")
            
            print("\n=== 数据读取完成 ===")
            print(f"成功读取的键: {list(data_dict.keys())}")
            print("\n请查看上述输出，确定正确的键名后，我们可以更新代码。\n")
            
        return data_dict




def create_units(data, t, RUL, num_units, len_units):
    """
    根据长度规格将原始数据组织成独立的单元。
    
    参数:
        data: 所有单元的输入特征
        t: 时间/循环信息
        RUL: 剩余使用寿命值
        num_units: 单元总数
        len_units: 每个单元的长度
        
    返回:
        data_list: 每个单元的数据数组列表
        RUL_list: 每个单元的RUL数组列表
    """
    # 将输入特征和时间/循环信息水平堆叠，创建完整的输入数据集
    data_all = np.hstack((data, t.flatten()[:, None]))
    # 保留所有RUL（剩余使用寿命）值
    RUL_all = RUL

    # 初始化用于存储每个单元数据的列表
    data_list = []
    # 初始化用于存储每个单元RUL值的列表
    RUL_list = []

    # 初始化起始索引为0
    idx_start = 0
    # 遍历所有单元
    for i in range(num_units):
        # 计算当前单元的结束索引
        idx_end = idx_start + int(len_units[i])
        # 从完整数据集中提取当前单元的数据片段并添加到列表
        data_list.append(data_all[idx_start:idx_end, :])
        # 从完整RUL数据中提取当前单元的RUL片段并添加到列表
        RUL_list.append(RUL_all[idx_start:idx_end, :])
        # 更新起始索引为下一个单元的开始位置
        idx_start += int(len_units[i])

    # 返回组织好的数据列表和RUL列表
    return data_list, RUL_list






def create_slices(data_units, RUL_units, seq_len_slices, steps_slices):
    """
    为训练神经网络创建重叠的数据切片。
    
    此函数从输入数据创建滑动窗口切片，用于基于序列的神经网络训练。
    
    参数:
        data_units: 每个单元的输入数据数组列表
        RUL_units: 每个单元的RUL数组列表
        seq_len_slices: 每个切片序列的长度
        steps_slices: 连续切片之间的步长
        
    返回:
        data_slices: 切片数据数组列表
        RUL_slices: 切片RUL数组列表
        num_slices: 每个单元的切片数量
    """
    # 初始化存储切片数据的列表
    data_slices = []
    # 初始化存储切片RUL数据的列表
    RUL_slices = []
    # 初始化每个单元切片数量的数组
    num_slices = np.zeros(len(data_units), dtype=np.int64)
    # 遍历所有数据单元
    for i in range(len(data_units)):
        # 计算当前单元可以创建的切片数量
        num_slices_tmp = int((data_units[i].shape[0] - max(seq_len_slices, steps_slices))
                             / steps_slices) + 1  # 每个单元的切片数量
        # 创建临时数组存储当前单元的数据切片
        data_slices_tmp = np.zeros(
            (num_slices_tmp, seq_len_slices, data_units[0].shape[1]))  # 每个单元的数据切片
        # 创建临时数组存储当前单元的RUL切片
        RUL_slices_tmp = np.zeros((num_slices_tmp, seq_len_slices, RUL_units[0].shape[1]))  # 每个单元的RUL切片
        # 初始化当前单元的起始索引
        idx_start = 0
        # 遍历当前单元的所有切片
        for j in range(num_slices_tmp):
            # 计算切片的结束索引
            idx_end = idx_start + seq_len_slices
            # 提取当前数据切片并存储
            data_slices_tmp[j, :, :] = data_units[i][idx_start:idx_end, :]
            # 提取当前RUL切片并存储
            RUL_slices_tmp[j, :, :] = RUL_units[i][idx_start:idx_end, :]
            # 更新起始索引，按步长移动
            idx_start += steps_slices
        # 将当前单元的数据切片添加到总列表
        data_slices.append(data_slices_tmp)
        # 将当前单元的RUL切片添加到总列表
        RUL_slices.append(RUL_slices_tmp)
        # 记录当前单元的切片数量
        num_slices[i] = num_slices_tmp
    # 返回所有数据切片、RUL切片和每个单元的切片数量
    return data_slices, RUL_slices, num_slices






def create_chosen_cells(data, idx_cells_train, idx_cells_test, perc_val):
    """
    为特定电池单元创建训练/验证/测试分割。
    
    此函数为训练和测试选择特定的电池单元，然后将训练单元分割为训练和验证集。
    
    参数:
        data: 包含所有数据的SeversonBattery对象
        idx_cells_train: 用于训练的单元索引列表
        idx_cells_test: 用于测试的单元索引列表
        perc_val: 用于验证的训练数据百分比
        
    返回:
        inputs: 包含训练/验证/测试输入张量的字典
        targets: 包含训练/验证/测试目标张量的字典
    """
    # 初始化训练、验证、测试输入切片列表
    inputs_train_slices = []
    inputs_val_slices = []
    inputs_test_slices = []
    # 初始化训练、验证、测试目标切片列表
    targets_train_slices = []
    targets_val_slices = []
    targets_test_slices = []

    # 遍历训练单元索引列表
    for idx in idx_cells_train:
        # 调整索引（MATLAB索引从1开始，Python从0开始）
        idx_true = idx - 1
        # 初始化临时输入和目标变量
        inputs_tmp = None
        targets_tmp = None
        # 检查当前索引是否在训练单元中
        if idx_true in data.idx_train_units:
            # 查找当前索引在训练单元中的位置
            idx_tmp = np.where(data.idx_train_units == idx_true)[0][0]
            # 获取对应的训练输入切片
            inputs_tmp = data.inputs_train_slices[idx_tmp]
            # 获取对应的目标切片
            targets_tmp = data.targets_train_slices[idx_tmp]
        # 检查当前索引是否在验证单元中
        elif idx_true in data.idx_val_units:
            # 查找当前索引在验证单元中的位置
            idx_tmp = np.where(data.idx_val_units == idx_true)[0][0]
            # 获取对应的验证输入切片
            inputs_tmp = data.inputs_val_slices[idx_tmp]
            # 获取对应的目标切片
            targets_tmp = data.targets_val_slices[idx_tmp]
        # 检查当前索引是否在测试单元中
        elif idx_true in data.idx_test_units:
            # 查找当前索引在测试单元中的位置
            idx_tmp = np.where(data.idx_test_units == idx_true)[0][0]
            # 获取对应的测试输入切片
            inputs_tmp = data.inputs_test_slices[idx_tmp]
            # 获取对应的目标切片
            targets_tmp = data.targets_test_slices[idx_tmp]
        # 如果输入和目标数据都有效
        if inputs_tmp is not None and targets_tmp is not None:
            # 使用train_test_split将数据分割为训练和验证部分
            inputs_tmp_train, inputs_tmp_val, targets_tmp_train, targets_tmp_val = train_test_split(
                inputs_tmp, targets_tmp,
                test_size=perc_val
            )
            # 将训练部分添加到训练输入列表
            inputs_train_slices.append(inputs_tmp_train)
            # 将验证部分添加到验证输入列表
            inputs_val_slices.append(inputs_tmp_val)
            # 将训练目标部分添加到训练目标列表
            targets_train_slices.append(targets_tmp_train)
            # 将验证目标部分添加到验证目标列表
            targets_val_slices.append(targets_tmp_val)

    # 遍历测试单元索引列表
    for idx in idx_cells_test:
        # 调整索引（MATLAB索引从1开始，Python从0开始）
        idx_true = idx - 1
        # 初始化临时输入和目标变量
        inputs_tmp = None
        targets_tmp = None
        # 检查当前索引是否在训练单元中
        if idx_true in data.idx_train_units:
            # 查找当前索引在训练单元中的位置
            idx_tmp = np.where(data.idx_train_units == idx_true)[0][0]
            # 获取对应的训练输入切片
            inputs_tmp = data.inputs_train_slices[idx_tmp]
            # 获取对应的目标切片
            targets_tmp = data.targets_train_slices[idx_tmp]
        # 检查当前索引是否在验证单元中
        elif idx_true in data.idx_val_units:
            # 查找当前索引在验证单元中的位置
            idx_tmp = np.where(data.idx_val_units == idx_true)[0][0]
            # 获取对应的验证输入切片
            inputs_tmp = data.inputs_val_slices[idx_tmp]
            # 获取对应的目标切片
            targets_tmp = data.targets_val_slices[idx_tmp]
        # 检查当前索引是否在测试单元中
        elif idx_true in data.idx_test_units:
            # 查找当前索引在测试单元中的位置
            idx_tmp = np.where(data.idx_test_units == idx_true)[0][0]
            # 获取对应的测试输入切片
            inputs_tmp = data.inputs_test_slices[idx_tmp]
            # 获取对应的目标切片
            targets_tmp = data.targets_test_slices[idx_tmp]
        # 如果输入和目标数据都有效
        if inputs_tmp is not None and targets_tmp is not None:
            # 将输入数据添加到测试输入列表
            inputs_test_slices.append(inputs_tmp)
            # 将目标数据添加到测试目标列表
            targets_test_slices.append(targets_tmp)

    # 将训练输入切片连接成单个数组
    inputs_train_ndarray = np.concatenate((inputs_train_slices), axis=0)
    # 将验证输入切片连接成单个数组
    inputs_val_ndarray = np.concatenate((inputs_val_slices), axis=0)
    # 将测试输入切片连接成单个数组
    inputs_test_ndarray = np.concatenate((inputs_test_slices), axis=0)
    # 将训练目标切片连接成单个数组
    targets_train_ndarray = np.concatenate((targets_train_slices), axis=0)
    # 将验证目标切片连接成单个数组
    targets_val_ndarray = np.concatenate((targets_val_slices), axis=0)
    # 将测试目标切片连接成单个数组
    targets_test_ndarray = np.concatenate((targets_test_slices), axis=0)

    # 将训练输入数组转换为torch张量
    inputs_train_tensor = torch.from_numpy(inputs_train_ndarray).type(torch.float32)
    # 将验证输入数组转换为torch张量
    inputs_val_tensor = torch.from_numpy(inputs_val_ndarray).type(torch.float32)
    # 将测试输入数组转换为torch张量
    inputs_test_tensor = torch.from_numpy(inputs_test_ndarray).type(torch.float32)
    # 将训练目标数组转换为torch张量
    targets_train_tensor = torch.from_numpy(targets_train_ndarray).type(torch.float32)
    # 将验证目标数组转换为torch张量
    targets_val_tensor = torch.from_numpy(targets_val_ndarray).type(torch.float32)
    # 将测试目标数组转换为torch张量
    targets_test_tensor = torch.from_numpy(targets_test_ndarray).type(torch.float32)

    # 创建输入字典
    inputs = dict()
    # 创建目标字典
    targets = dict()

    # 将训练输入张量添加到输入字典
    inputs['train'] = inputs_train_tensor
    # 将验证输入张量添加到输入字典
    inputs['val'] = inputs_val_tensor
    # 将测试输入张量添加到输入字典
    inputs['test'] = inputs_test_tensor

    # 将训练目标张量添加到目标字典
    targets['train'] = targets_train_tensor
    # 将验证目标张量添加到目标字典
    targets['val'] = targets_val_tensor
    # 将测试目标张量添加到目标字典
    targets['test'] = targets_test_tensor

    # 返回输入和目标字典
    return inputs, targets






def standardize_tensor(data, mode, mean=0, std=1):
    """
    通过去除均值和缩放到单位方差来标准化张量数据。
    
    此函数对输入张量执行z-score归一化。
    
    参数:
        data: 要标准化的输入张量
        mode: 'fit'从数据计算均值和标准差，或'transform'使用提供的值
        mean: 用于标准化的均值（如果mode != 'fit'）
        std: 用于标准化的标准差（如果mode != 'fit'）
        
    返回:
        data_norm: 标准化张量
        mean: 用于标准化的均值
        std: 用于标准化的标准差
    """
    data_2D = data.contiguous().view((-1, data.shape[-1]))  # 重塑为2D
    if mode == 'fit':
        mean = torch.mean(data_2D, dim=0)
        std = torch.std(data_2D, dim=0)
    data_norm_2D = (data_2D - mean) / (std + 1e-8)
    data_norm = data_norm_2D.contiguous().view((-1, data.shape[-2], data.shape[-1]))
    return data_norm, mean, std






def inverse_standardize_tensor(data_norm, mean, std):
    """
    逆标准化，将归一化数据转换回原始尺度。
    
    参数:
        data_norm: 标准化张量
        mean: 原始标准化使用的均值
        std: 原始标准化使用的标准差
        
    返回:
        data: 原始尺度的张量
    """
    data_norm_2D = data_norm.contiguous().view((-1, data_norm.shape[-1]))  # 重塑为2D
    data_2D = data_norm_2D * std + mean
    data = data_2D.contiguous().view((-1, data_norm.shape[-2], data_norm.shape[-1]))
    return data






def Verhulst(y, r, K, C):
    """
    用于逻辑增长建模的Verhulst函数。
    
    此函数表示Verhulst逻辑增长模型，通常用于电池退化建模。
    
    参数:
        y: 当前值
        r: 增长率参数
        K: 环境容纳量参数
        C: 偏移参数
        
    返回:
        逻辑增长速率
    """
    return r * (y - C) * (1 - (y - C) / (K - C))






class Sin(nn.Module):
    """
    实现正弦激活函数的自定义PyTorch模块。
    
    此激活函数可以在神经网络中用作tanh或ReLU等标准激活函数的替代。
    """
    def forward(self, input):
        return torch.sin(input)






class Neural_Net(nn.Module):
    """
    具有可定制架构的基础前馈神经网络。
    
    此类创建具有指定层数和激活函数的全连接神经网络。
    包括用于正则化的dropout层。
    
    参数:
        seq_len: 输入序列的长度
        inputs_dim: 输入特征的维度
        outputs_dim: 输出特征的维度
        layers: 指定每个隐藏层中神经元数量的列表
        activation: 要使用的激活函数（'Tanh'或'Sin'）
    """
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, activation='Tanh'):
        super(Neural_Net, self).__init__()

        self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim

        self.layers = []

        self.layers.append(nn.Linear(in_features=inputs_dim, out_features=layers[0]))
        nn.init.xavier_normal_(self.layers[-1].weight)


        if activation == 'Tanh':
            self.layers.append(nn.Tanh())
        elif activation == 'Sin':
            self.layers.append(Sin())
        self.layers.append(nn.Dropout(p=0.2))


        for l in range(len(layers) - 1):
            self.layers.append(nn.Linear(in_features=layers[l], out_features=layers[l + 1]))
            nn.init.xavier_normal_(self.layers[-1].weight)

            if activation == 'Tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'Sin':
                self.layers.append(Sin())
            self.layers.append(nn.Dropout(p=0.2))

        if len(layers) > 0:
            self.layers.append(nn.Linear(in_features=layers[l + 1], out_features=outputs_dim))
        else:
            # If no hidden layers, connect input directly to output
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






class DataDrivenNN(nn.Module):
    """
    用于电池状态估算的数据驱动神经网络。
    
    这是一个没有物理约束的纯神经网络模型。
    它通过代理神经网络处理输入并返回电池状态的预测。
    
    参数:
        seq_len: 输入序列的长度
        inputs_dim: 输入特征的维度
        outputs_dim: 输出特征的维度
        layers: 指定每个隐藏层中神经元数量的列表
        scaler_inputs: 输入归一化参数（均值，标准差）
        scaler_targets: 目标归一化参数（均值，标准差）
    """
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, scaler_inputs, scaler_targets):
        # 调用父类构造函数
        super(DataDrivenNN, self).__init__()
        # 保存序列长度、输入维度和输出维度
        self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim
        # 保存输入和目标的归一化参数
        self.scaler_inputs, self.scaler_targets = scaler_inputs, scaler_targets

        # 创建代理神经网络
        self.surrogateNN = Neural_Net(
            seq_len=self.seq_len,
            inputs_dim=self.inputs_dim,
            outputs_dim=self.outputs_dim,
            layers=layers
        )

    def forward(self, inputs):
        # 提取状态特征（除循环次数外的所有特征）
        s = inputs[:, :, 0: self.inputs_dim - 1]
        # 提取循环次数特征（最后一个维度）
        t = inputs[:, :, self.inputs_dim - 1:]
        # 设置循环次数张量需要计算梯度
        t.requires_grad_(True)

        # 对状态特征进行归一化
        s_norm, _, _ = standardize_tensor(s, mode='transform', mean=self.scaler_inputs[0][0: self.inputs_dim - 1],
                                          std=self.scaler_inputs[1][0: self.inputs_dim - 1])

        # 对循环次数特征进行归一化
        t_norm, _, _ = standardize_tensor(t, mode='transform', mean=self.scaler_inputs[0][self.inputs_dim - 1:],
                                          std=self.scaler_inputs[1][self.inputs_dim - 1:])

        # 通过代理神经网络预测归一化的电池状态
        U_norm = self.surrogateNN(x=torch.cat((s_norm, t_norm), dim=2))
        # 将预测结果反归一化到原始尺度
        U = inverse_standardize_tensor(U_norm, mean=self.scaler_targets[0], std=self.scaler_targets[1])

        # 创建梯度计算的输出张量
        grad_outputs = torch.ones_like(U)
        # 计算电池状态对循环次数的梯度
        U_t = torch.autograd.grad(
            U, t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 初始化物理约束项为零
        F = torch.zeros_like(U)
        # 初始化物理约束项的循环次数导数为零
        F_t = torch.zeros_like(U)

        # 保存循环次数导数
        self.U_t = U_t

        # 返回预测的电池状态、物理约束项及其循环次数导数
        return U, F, F_t






class VerhulstPINN(nn.Module):
    """
    使用Verhulst退化模型的物理信息神经网络(PINN)。
    
    此模型在训练期间将Verhulst逻辑增长模型作为物理约束纳入。
    它在拟合数据的同时学习Verhulst方程的参数。
    
    参数:
        seq_len: 输入序列的长度
        inputs_dim: 输入特征的维度
        outputs_dim: 输出特征的维度
        layers: 指定每个隐藏层中神经元数量的列表
        scaler_inputs: 输入归一化参数（均值，标准差）
        scaler_targets: 目标归一化参数（均值，标准差）
    """
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, scaler_inputs, scaler_targets):
        # 调用父类构造函数
        super(VerhulstPINN, self).__init__()
        # 保存序列长度、输入维度和输出维度
        self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim
        # 保存输入和目标的归一化参数
        self.scaler_inputs, self.scaler_targets = scaler_inputs, scaler_targets

        # 初始化可学习参数log_p_r，用于控制增长率r
        self.log_p_r = torch.nn.Parameter(torch.randn(()), requires_grad=True)
        # 初始化可学习参数log_p_K，用于控制环境容纳量K
        self.log_p_K = torch.nn.Parameter(torch.randn(()), requires_grad=True)
        # 初始化可学习参数log_p_C，用于控制偏移量C
        self.log_p_C = torch.nn.Parameter(torch.randn(()), requires_grad=True)

        # 设置环境容纳量K的下界
        self.lb_p_K = 0.2
        # 设置环境容纳量K的上界
        self.ub_p_K = 1.

        # 设置偏移量C的下界
        self.lb_p_C = 0.
        # 设置偏移量C的上界
        self.ub_p_C = 0.1

        # 创建代理神经网络
        self.surrogateNN = Neural_Net(
            seq_len=self.seq_len,
            inputs_dim=self.inputs_dim,
            outputs_dim=self.outputs_dim,
            layers=layers
        )

    @property
    def p_r(self):
        # 通过指数变换确保增长率r为正数
        return torch.exp(-self.log_p_r)

    @property
    def p_K(self):
        # 通过sigmoid函数和边界限制确保环境容纳量K在指定范围内
        return self.lb_p_K + (self.ub_p_K - self.lb_p_K) * torch.sigmoid(self.log_p_K)

    @property
    def p_C(self):
        # 通过sigmoid函数和边界限制确保偏移量C在指定范围内
        return self.lb_p_C + (self.ub_p_C - self.lb_p_C) * torch.sigmoid(self.log_p_C)

    def forward(self, inputs):
        # 提取状态特征（除时间外的所有特征）
        s = inputs[:, :, 0: self.inputs_dim - 1]
        # 提取时间特征（最后一个维度）
        t = inputs[:, :, self.inputs_dim - 1:]

        # 对状态特征进行归一化
        s_norm, _, _ = standardize_tensor(s, mode='transform', mean=self.scaler_inputs[0][0: self.inputs_dim - 1],
                                          std=self.scaler_inputs[1][0: self.inputs_dim - 1])
        # 设置时间张量需要计算梯度
        t.requires_grad_(True)
        # 对时间特征进行归一化
        t_norm, _, _ = standardize_tensor(t, mode='transform', mean=self.scaler_inputs[0][self.inputs_dim - 1:],
                                          std=self.scaler_inputs[1][self.inputs_dim - 1:])
        # 设置归一化后的时间张量需要计算梯度
        t_norm.requires_grad_(True)

        # 通过代理神经网络预测归一化的电池状态
        U_norm = self.surrogateNN(x=torch.cat((s_norm, t_norm), dim=2))
        # 将预测结果反归一化到原始尺度
        U = inverse_standardize_tensor(U_norm, mean=self.scaler_targets[0], std=self.scaler_targets[1])

        # 创建梯度计算的输出张量
        grad_outputs = torch.ones_like(U)
        # 计算电池状态对时间的梯度
        U_t = torch.autograd.grad(
            U, t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 计算Verhulst模型的输出
        G = Verhulst(y=U, r=self.p_r, K=self.p_K, C=self.p_C)

        # 计算物理约束项（时间导数与Verhulst模型输出的差值）
        F = U_t - G
        # 计算物理约束项的时间导数
        F_t = torch.autograd.grad(
            F, t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 保存时间导数
        self.U_t = U_t
        # 返回预测的电池状态、物理约束项及其时间导数
        return U, F, F_t






class DeepHPMNN(nn.Module):
    """
    深度混合物理-经验模型神经网络。
    
    此模型将用于状态预测的代理神经网络与建模基础物理的动态神经网络相结合。
    
    参数:
        seq_len: 输入序列的长度
        inputs_dim: 输入特征的维度
        outputs_dim: 输出特征的维度
        layers: 指定每个隐藏层中神经元数量的列表
        scaler_inputs: 输入归一化参数（均值，标准差）
        scaler_targets: 目标归一化参数（均值，标准差）
        inputs_dynamical: 动态模型的输入规格
        inputs_dim_dynamical: 动态模型输入的维度
    """
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, scaler_inputs, scaler_targets,
                 inputs_dynamical, inputs_dim_dynamical):
        # 调用父类构造函数
        super(DeepHPMNN, self).__init__()
        # 保存序列长度、输入维度和输出维度
        self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim
        # 保存输入和目标的归一化参数
        self.scaler_inputs, self.scaler_targets = scaler_inputs, scaler_targets

        # 处理动态模型的输入规格
        if len(inputs_dynamical.split(',')) <= 1:
            # 如果输入规格只包含一个变量，则直接使用
            self.inputs_dynamical = inputs_dynamical
        else:
            # 如果输入规格包含多个变量，则使用torch.cat连接
            self.inputs_dynamical = 'torch.cat((' + inputs_dynamical + '), dim=2)'
        # 计算动态模型输入的维度
        self.inputs_dim_dynamical = eval(inputs_dim_dynamical)

        # 创建用于状态预测的代理神经网络
        self.surrogateNN = Neural_Net(
            seq_len=self.seq_len,
            inputs_dim=self.inputs_dim,
            outputs_dim=self.outputs_dim,
            layers=layers
        )

        # 创建用于建模物理动态的神经网络
        self.dynamicalNN = Neural_Net(
            seq_len=self.seq_len,
            inputs_dim=self.inputs_dim_dynamical,
            outputs_dim=1,
            layers=layers
        )

    def forward(self, inputs):
        # 提取状态特征（除时间外的所有特征）
        s = inputs[:, :, 0: self.inputs_dim - 1]
        # 提取时间特征（最后一个维度）
        t = inputs[:, :, self.inputs_dim - 1:]

        # 设置状态张量需要计算梯度
        s.requires_grad_(True)
        # 对状态特征进行归一化
        s_norm, _, _ = standardize_tensor(s, mode='transform', mean=self.scaler_inputs[0][0: self.inputs_dim - 1],
                                          std=self.scaler_inputs[1][0: self.inputs_dim - 1])
        # 设置归一化后的状态张量需要计算梯度
        s_norm.requires_grad_(True)

        # 设置时间张量需要计算梯度
        t.requires_grad_(True)
        # 对时间特征进行归一化
        t_norm, _, _ = standardize_tensor(t, mode='transform', mean=self.scaler_inputs[0][self.inputs_dim - 1:],
                                          std=self.scaler_inputs[1][self.inputs_dim - 1:])
        # 设置归一化后的时间张量需要计算梯度
        t_norm.requires_grad_(True)

        # 通过代理神经网络预测归一化的电池状态
        U_norm = self.surrogateNN(x=torch.cat((s_norm, t_norm), dim=2))
        # 将预测结果反归一化到原始尺度
        U = inverse_standardize_tensor(U_norm, mean=self.scaler_targets[0], std=self.scaler_targets[1])

        # 创建梯度计算的输出张量
        grad_outputs = torch.ones_like(U)
        # 计算电池状态对时间的梯度
        U_t = torch.autograd.grad(
            U, t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 计算电池状态对状态特征的梯度
        U_s = torch.autograd.grad(
            U, s,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 评估动态神经网络的输出
        G = eval('self.dynamicalNN(x=' + self.inputs_dynamical + ')')

        # 计算物理约束项（时间导数与动态模型输出的差值）
        F = U_t - G
        # 计算物理约束项的时间导数
        F_t = torch.autograd.grad(
            F, t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 保存时间导数
        self.U_t = U_t
        # 返回预测的电池状态、物理约束项及其时间导数
        return U, F, F_t






class TensorDataset(Dataset):
    """
    用于处理张量对的自定义PyTorch数据集。
    
    此类将输入和目标张量包装在数据集中，可以与PyTorch DataLoader
    一起使用进行批量训练。
    
    参数:
        data_tensor: 输入张量
        target_tensor: 目标张量
    """
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)






class My_loss(nn.Module):
    """
    用于不同训练模式的自定义损失函数。
    
    此损失函数支持不同的模式：
    - 'Baseline': 标准数据拟合损失
    - 'Sum': 包含物理约束
    - 'AdpBal': 不同损失项的自适应平衡
    
    参数:
        mode: 训练模式（'Baseline'、'Sum'或'AdpBal'）
    """
    def __init__(self, mode):
        # 调用父类构造函数
        super().__init__()
        # 保存训练模式
        self.mode = mode

    def forward(self, outputs_U, targets_U, outputs_F, outputs_F_t, log_sigma_u, log_sigma_f, log_sigma_f_t):
        # 计算数据拟合损失：预测输出与真实目标之间的均方误差
        loss_U = torch.sum((outputs_U - targets_U) ** 2)

        # 计算物理约束损失：物理方程残差的平方和
        loss_F = torch.sum(outputs_F ** 2)

        # 计算物理约束时间导数损失：物理方程残差循环次数导数的平方和
        loss_F_t = torch.sum((outputs_F_t) ** 2)

        # 根据指定的训练模式计算总损失
        if self.mode == 'Baseline':
            # 基线模式：仅使用数据拟合损失
            loss = loss_U
        elif self.mode == 'Sum':
            # 求和模式：数据拟合损失 + 物理约束损失 + 物理约束时间导数损失
            loss = loss_U + loss_F + loss_F_t
        elif self.mode == 'AdpBal':
            # 自适应平衡模式：使用可学习的不确定性参数来平衡不同损失项
            # 通过指数变换确保权重为正数
            loss = torch.exp(-log_sigma_u) * loss_U + torch.exp(-log_sigma_f) * loss_F + torch.exp(-log_sigma_f_t) * loss_F_t + \
                   log_sigma_u + log_sigma_f + log_sigma_f_t
        else:
            # 如果没有匹配的模式，使用基线模式
            loss = loss_U
        # print(' Loss_U: {:.5f}, Loss_F: {:.5f},'.format(loss_U, loss_F))

        # 保存各个损失项，便于后续分析和监控
        self.loss_U = loss_U
        self.loss_F = loss_F
        self.loss_F_t = loss_F_t
        # 返回计算得到的总损失
        return loss






def train(num_epoch, batch_size, train_loader, num_slices_train, inputs_val, targets_val,
          model, optimizer, scheduler, criterion, log_sigma_u, log_sigma_f, log_sigma_f_t):
    """
    神经网络模型的训练函数。
    
    此函数执行训练循环，包括前向传播、损失计算、反向传播和验证评估。
    
    参数:
        num_epoch: 训练轮数
        batch_size: 训练批次大小
        train_loader: 训练数据的DataLoader
        num_slices_train: 训练数据切片数量
        inputs_val: 验证输入
        targets_val: 验证目标
        model: 要训练的神经网络模型
        optimizer: 优化算法
        scheduler: 学习率调度器
        criterion: 损失函数
        log_sigma_u: U的不确定性参数的对数
        log_sigma_f: F的不确定性参数的对数
        log_sigma_f_t: F_t的不确定性参数的对数
        
    返回:
        model: 训练后的模型
        results_epoch: 每个epoch的训练结果
    """
    # 计算每个epoch中的训练批次数量
    num_period = int(num_slices_train / batch_size)
    # 初始化epoch级结果字典
    results_epoch = dict()
    # 初始化训练损失记录
    results_epoch['loss_train'] = torch.zeros(num_epoch)
    # 初始化验证损失记录
    results_epoch['loss_val'] = torch.zeros(num_epoch)
    # 初始化参数p_r记录
    results_epoch['p_r'] = torch.zeros(num_epoch)
    # 初始化参数p_K记录
    results_epoch['p_K'] = torch.zeros(num_epoch)
    # 初始化参数p_C记录
    results_epoch['p_C'] = torch.zeros(num_epoch)
    # 初始化U项方差记录
    results_epoch['var_U'] = torch.zeros(num_epoch)
    # 初始化F项方差记录
    results_epoch['var_F'] = torch.zeros(num_epoch)
    # 初始化F_t项方差记录
    results_epoch['var_F_t'] = torch.zeros(num_epoch)

    # 遍历每个训练轮次
    for epoch in range(num_epoch):
        # 设置模型为训练模式
        model.train()
        # 初始化周期级结果字典
        results_period = dict()
        # 初始化周期训练损失记录
        results_period['loss_train'] = torch.zeros(num_period)
        # 初始化周期参数p_r记录
        results_period['p_r'] = torch.zeros(num_period)
        # 初始化周期参数p_K记录
        results_period['p_K'] = torch.zeros(num_period)
        # 初始化周期参数p_C记录
        results_period['p_C'] = torch.zeros(num_period)
        # 初始化周期U项方差记录
        results_period['var_U'] = torch.zeros(num_period)
        # 初始化周期F项方差记录
        results_period['var_F'] = torch.zeros(num_period)
        # 初始化周期F_t项方差记录
        results_period['var_F_t'] = torch.zeros(num_period)
        # 禁用cuDNN以确保可重现性
        with torch.backends.cudnn.flags(enabled=False):
            # 遍历训练数据加载器中的每个批次
            for period, (inputs_train_batch, targets_train_batch) in enumerate(train_loader):
                
                # 前向传播：获取模型预测结果
                U_pred_train, F_pred_train, F_t_pred_train = model(inputs=inputs_train_batch)
                # 计算损失
                loss = criterion(
                    outputs_U=U_pred_train,
                    targets_U=targets_train_batch,
                    outputs_F=F_pred_train,
                    outputs_F_t=F_t_pred_train,
                    log_sigma_u=log_sigma_u,
                    log_sigma_f=log_sigma_f,
                    log_sigma_f_t=log_sigma_f_t
                )

                # 清零优化器梯度
                optimizer.zero_grad()
                # 反向传播计算梯度
                loss.backward()
                # 更新模型参数
                optimizer.step()

                # 记录数据拟合损失
                results_period['loss_train'][period] = criterion.loss_U.detach()
                # 尝试记录物理参数
                try:
                    results_period['p_r'][period] = model.p_r.detach()
                    results_period['p_K'][period] = model.p_K.detach()
                    results_period['p_C'][period] = model.p_C.detach()
                except:
                    # 如果模型没有这些参数，则跳过
                    pass
                # 记录U项的方差
                results_period['var_U'][period] = torch.exp(-log_sigma_u).detach()
                # 记录F项的方差
                results_period['var_F'][period] = torch.exp(-log_sigma_f).detach()
                # 记录F_t项的方差
                results_period['var_F_t'][period] = torch.exp(-log_sigma_f_t).detach()

                # 每个周期打印训练信息
                if (epoch + 1) % 1 == 0 and (period + 1) % 1 == 0:  # 每 100 次输出结果
                    print(
                        'Epoch: {}, Period: {}, Loss: {:.5f}, Loss_U: {:.5f}, Loss_F: {:.5f}, Loss_F_t: {:.5f}'.format(
                            epoch + 1, period + 1, loss, criterion.loss_U, criterion.loss_F, criterion.loss_F_t))

        # 计算当前epoch的平均训练损失
        results_epoch['loss_train'][epoch] = torch.mean(results_period['loss_train'])
        # 计算当前epoch的平均p_r参数
        results_epoch['p_r'][epoch] = torch.mean(results_period['p_r'])
        # 计算当前epoch的平均p_K参数
        results_epoch['p_K'][epoch] = torch.mean(results_period['p_K'])
        # 计算当前epoch的平均p_C参数
        results_epoch['p_C'][epoch] = torch.mean(results_period['p_C'])
        # 计算当前epoch的平均U项方差
        results_epoch['var_U'][epoch] = torch.mean(results_period['var_U'])
        # 计算当前epoch的平均F项方差
        results_epoch['var_F'][epoch] = torch.mean(results_period['var_F'])
        # 计算当前epoch的平均F_t项方差
        results_epoch['var_F_t'][epoch] = torch.mean(results_period['var_F_t'])

        # 设置模型为评估模式
        model.eval()
        # 在验证集上进行前向传播
        U_pred_val, F_pred_val, F_t_pred_val = model(inputs=inputs_val)
        # 计算验证损失
        loss_val = criterion(
            outputs_U=U_pred_val,
            targets_U=targets_val,
            outputs_F=F_pred_val,
            outputs_F_t=F_t_pred_val,
            log_sigma_u=log_sigma_u,
            log_sigma_f=log_sigma_f,
            log_sigma_f_t=log_sigma_f_t
        )
        # 更新学习率
        scheduler.step()
        # 记录验证数据拟合损失
        results_epoch['loss_val'][epoch] = criterion.loss_U.detach()

    # 返回训练完成的模型和训练结果
    return model, results_epoch

