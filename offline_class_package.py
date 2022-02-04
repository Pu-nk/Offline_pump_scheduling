'''
yw pred model using GRU model
Author:Pu Zhengheng
'''

import os
import joblib
from torch import optim
from collections import namedtuple
from torch.utils.data import Dataset
from alive_progress import alive_bar, config_handler
from interval import Interval
from time import ctime
from gen_context import *
from duration_model import get_input, linear_reg
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
## 全局参数定义
input_size = 7  # 模型输入特征维度


# -------------
# 模型结构定义:GRU模型
# -------------
class BaseModel(nn.Module):
    def __init__(self, input_size=36, hidden_units_1=32, hidden_units_2=8, layer_num=1, type='tank', cell='GRU'):

        super(BaseModel, self).__init__()
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.input_size = input_size
        self.layer_num = layer_num
        self.type = type
        if cell == 'LSTM':
            self.cell = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_units,
                                num_layers=self.layer_num, dropout=0.0, batch_first=True)
        if cell == 'GRU':
            self.cell = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_units_1,
                               num_layers=self.layer_num, dropout=0.0, batch_first=True)
            if self.type == 'water_level':
                # 子网络的输入维度
                sec_size = 1
            elif self.type == 'pressure':
                # 子网络的输入维度
                sec_size = 5
            else:
                raise ValueError

            self.deep_cell = nn.GRU(input_size=sec_size, hidden_size=self.hidden_units_2,
                                    num_layers=self.layer_num, dropout=0.0, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_units_1 + self.hidden_units_2, 4)
        self.fc2 = nn.Linear(4, 1)


# ---------------
# 门控单元循环网络
# ---------------
class GRU_Model(BaseModel):

    def __init__(self, device, input_size, hidden_units_1, hidden_units_2, layer_num=1, type='water_level', cell='GRU'):
        super(GRU_Model, self).__init__(input_size, hidden_units_1, hidden_units_2, layer_num, type, cell)
        self.length = 36
        self.steps = 5
        self.device = device
        self.reshape = lambda x, hn, units: hn.view(x.size(0), units)

    def forward(self, x):
        # 进水压力、测压点压力、时间编码特征（36*3*1）
        x1 = x[:, :self.length * self.input_size].view(-1, self.length, self.input_size)
        # 前五日最大同时刻液位数据特征（5*1*1）
        if self.type == 'water_level':
            x2 = torch.max(x[:, self.input_size * self.length:].view(-1, self.steps, 5), axis=2)[0].view(
                -1, self.steps, 1)
        # 前五日同时刻压力数据特征（5*5*1）
        elif self.type == 'pressure':
            x2 = x[:, self.input_size * self.length:].view(-1, self.steps, 5)
        else:
            raise ValueError

        hn1 = self.cell(x1)[-1]
        hn2 = self.deep_cell(x2)[-1]
        hn1 = self.reshape(x1, hn1, self.hidden_units_1)
        hn2 = self.reshape(x1, hn2, self.hidden_units_2)
        hn = torch.cat((hn1, hn2), dim=1)
        fcOutput = F.relu(self.fc1(hn))
        fcOutput = self.fc2(fcOutput)
        return fcOutput


'''
数据预处理模，时间序列样本的生成、训练集和测试集的划分
以及Pytorch数据集格式的封装
'''


# ----------------------
# 定义 Dataset
# ----------------------
class Time_Series_Data(Dataset):
    def __init__(self, train_x, train_y):
        self.X = train_x
        self.y = train_y

    def __getitem__(self, item):
        x_t = self.X[item]
        y_t = self.y[item]
        return x_t, y_t

    def __len__(self):
        return len(self.X)


# ----------------------
# 数据预处理模块
# ----------------------
class Data_Preparement:
    '''
    input: DataFrame which contains all features model needs
    output: the train Dataset normalized by Minmaxscalar
    '''

    def __init__(self, station, type, data, size, n_out=1, trans=None, denose=False):
        self.station = station
        self.type = type
        self.denose = denose
        self.data = data.copy()  # 使用copy方法避免篡改原始数据
        self.size = size  # 输入步长
        self.n_out = n_out  # 预测步长
        self.day_step = 288  # 五分钟数据集的日步长
        self.data['time_encode'] = [i.hour * 60 + i.minute for i in self.data.index.time]  # 计算时间特征
        if trans:
            self.data[self.data.columns] = trans.fit_transform(self.data)  # 归一化数据
        self.sample = self._CreateSample()
        self.dataSet = namedtuple('dataSet', ['x', 'y'])
        self.valnum = self.day_step * 3  # 取3天作为验证集
        self.train = self._DivideTrainTest()[0]  # 选取训练集
        self.val = self._DivideTrainTest()[1]  # 选取验证集

    # 修改数据集的维度以符合模型输入条件，此处固定axis=2
    def _unsqeeze(self, axis, *data_group):
        res = list()
        for data in data_group:
            size = [data.x.shape[0], data.x.shape[1]]
            size.insert(axis, 1)
            temp = data.x.reshape(size)
            data = data._replace(x=temp)
            res.append(data)
        return res

    # 创建样本
    def _CreateSample(self):
        cols = list()
        ## 提取单一特征数据
        # 时间编码数据
        time_encode = self.data[['time_encode']]
        # 测压点压力
        bottom_pressure = self.data[['bottom_pressure']]
        # 水厂流量
        water_demand = self.data[['water_demand']]
        # 水厂压力
        xj_pressure = self.data[['xj_pressure']]
        # 根据预测的站点名选择出水压力、液位数据
        if self.station == 'hx':
            # 出水压力数据
            pump_pressure = self.data[['hx_pressure']]
            alt_pump_pressure = self.data[['xfx_pressure']]
            # 液位数据
            water_level = self.data[['hx_water_level']]
            alt_water_level = self.data[['xfx_water_level']]
        elif self.station == 'xfx':
            # 出水压力数据
            pump_pressure = self.data[['xfx_pressure']]
            alt_pump_pressure = self.data[['hx_pressure']]
            # 液位数据
            water_level = self.data[['xfx_water_level']]
            alt_water_level = self.data[['hx_water_level']]
        else:
            raise ValueError

        # 取前三小时数据
        for i in range(self.size - 1, -1, -1):
            cols.append(bottom_pressure.shift(i))
        for i in range(self.size - 1, -1, -1):
            cols.append(time_encode.shift(i))
        if self.type == 'water_level':
            # 加入前泵站出站压力特征
            for i in range(self.size - 1, -1, -1):
                cols.append(pump_pressure.shift(i))
            for i in range(self.size - 1, -1, -1):
                cols.append(alt_pump_pressure.shift(i))
            for i in range(self.size - 1, -1, -1):
                cols.append(water_demand.shift(i))
            for i in range(self.size - 1, -1, -1):
                cols.append(xj_pressure.shift(i))
            for i in range(self.size - 1, -1, -1):
                cols.append(alt_water_level.shift(i))
            # 加入前三天同时刻的特征
            for i in list(range(5 * self.day_step + 2, 5 * self.day_step - 3, -1)) + \
                     list(range(4 * self.day_step + 2, 4 * self.day_step - 3, -1)) + \
                     list(range(3 * self.day_step + 2, 3 * self.day_step - 3, -1)) + \
                     list(range(2 * self.day_step + 2, 2 * self.day_step - 3, -1)) + \
                     list(range(self.day_step + 2, self.day_step - 3, -1)):
                cols.append(water_level.shift(i))
            cols.append(water_level.shift(-self.n_out))
        elif self.type == 'pressure':
            # 加入前泵站液位特征
            for i in range(self.size - 1, -1, -1):
                cols.append(water_level.shift(i))
            for i in range(self.size - 1, -1, -1):
                cols.append(alt_water_level.shift(i))
            for i in range(self.size - 1, -1, -1):
                cols.append(water_demand.shift(i))
            for i in range(self.size - 1, -1, -1):
                cols.append(xj_pressure.shift(i))
            for i in range(self.size - 1, -1, -1):
                cols.append(alt_pump_pressure.shift(i))
            # 加入前三天同时刻的特征
            for i in list(range(5 * self.day_step + 2, 5 * self.day_step - 3, -1)) + \
                     list(range(4 * self.day_step + 2, 4 * self.day_step - 3, -1)) + \
                     list(range(3 * self.day_step + 2, 3 * self.day_step - 3, -1)) + \
                     list(range(2 * self.day_step + 2, 2 * self.day_step - 3, -1)) + \
                     list(range(self.day_step + 2, self.day_step - 3, -1)):
                cols.append(pump_pressure.shift(i))
            cols.append(pump_pressure.shift(-self.n_out))
        sample = pd.concat(cols, axis=1)
        sample.dropna(inplace=True)
        return sample.values

    # 划分样本和标签，训练集和验证集
    def _DivideTrainTest(self):
        split = lambda x: self.dataSet(x[:, :-1], np.squeeze(x[:, -1:]))
        train, val = self.sample[:-self.valnum], self.sample[-self.valnum:]
        train, val = map(split, [train, val])
        train, val = self._unsqeeze(2, train, val)
        train = Time_Series_Data(train.x, train.y)
        val = Time_Series_Data(val.x, val.y)
        return train, val


'''
定义神经网络训练模块
input: DataFrame格式的训练数据(包括液位、青东测压点压力、华翔进水压力)
output: 存储验证集上误差最小的模型参数到./yw_model, 存储归一化数据的scalar到./yw_scalar
'''


# ----------------------
# 定义早停类
# ----------------------
class EarlyStopping():
    def __init__(self, patience=15, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, bar, label):
        # print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, bar, label)
        elif score < self.best_score + self.delta:
            self.counter += 1
            bar.text(f'EarlyStopping counter:{self.counter} out of {self.patience}')
            bar()
            # print(f'EarlyStopping counter:{self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, bar, label)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, bar, label):
        if self.verbose:
            bar.text(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            bar()
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            torch.save(model.state_dict(), os.path.join(path, 'model_best_{}.pth'.format(label)))
            self.val_loss_min = val_loss


# ----------------------
# 定义模型训练类
# ----------------------
class pump_model_train:
    # 载入模型参数
    def __init__(self, station_name, obj_type):
        self.station_name = station_name  # 站点名称
        self.obj_type = obj_type  # 预测目标
        self.input_size = 36  # 样本大小
        self.batch_size = 128  # 小批次规模
        self.max_epoch = 100  # 最大训练代数
        self.lr = 0.0005  # 初始学习率
        self.trans = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练采用的硬件
        self.optim_p = optim.Adam  # 定义优化器
        self.loss_f = nn.MSELoss()  # 定义损失函数
        self.scalar_path = './scalar_pump'
        self.model_path = './{}'.format('_'.join([self.station_name, self.obj_type]))  # 定义模型储存路径
        if not os.path.exists(self.model_path):  # 如果目录不存在创建目录
            os.mkdir(self.model_path)
        # 创建网络
        self.net = GRU_Model(device=self.device, input_size=input_size, hidden_units_1=32, hidden_units_2=8,
                             type=self.obj_type)

    # 定义单个epoch的训练过程
    def _train(self, train_loader, optimizer):
        self.net.train()  # 启动训练
        epochloss = 0.0  # 用于记录一个epoch的损失
        for batch_idx, (inp, label) in enumerate(train_loader):
            inp, label = inp.to(device=self.device, dtype=torch.float), label.to(device=self.device,
                                                                                 dtype=torch.float)  # 将输入导入硬件
            optimizer.zero_grad()  # 梯度置0
            out = self.net.forward(inp)  # 前向传播
            out = out.squeeze()
            loss = self.loss_f(out, label)  # 计算损失函数
            epochloss += loss.item()  # 记录损失
            loss.backward()  # 梯度反向传播
            optimizer.step()  # 梯度更新
        return epochloss / len(train_loader.dataset)

    # 定义单个epoch的验证过程
    def _val(self, val_loader):
        self.net.eval()  # 启动验证
        val_loss = 0.0
        # 逐次验证
        with torch.no_grad():
            for inp, label in val_loader:
                inp, label = inp.to(device=self.device, dtype=torch.float), label.to(device=self.device,
                                                                                     dtype=torch.float)
                out = self.net.forward(inp)
                out = out.squeeze()
                val_loss += self.loss_f(out, label).item()  # 计算验证集损失
        return val_loss / len(val_loader.dataset)

    def epoches_train(self, train_data):
        '''
        :param train_data: 训练数据
        :return:
        '''
        self._setup_seed()
        self.net = self.net.to(self.device)
        time_label = train_data.index[-1].strftime("%Y_%m_%d")
        Dataset = Data_Preparement(self.station_name, self.obj_type, train_data, size=self.input_size, trans=self.trans)
        # 加载数据集
        train_loader = torch.utils.data.DataLoader(Dataset.train, batch_size=self.batch_size, shuffle=True,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(Dataset.val, batch_size=self.batch_size, shuffle=False,
                                                 drop_last=True)
        # 保存scalar
        if not os.path.exists(self.scalar_path):
            os.mkdir(self.scalar_path)
        joblib.dump(self.trans, '{}/scalar_{}'.format(self.scalar_path, time_label))

        # 统计模型参数数量
        trainable_param_n = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('Number of learnable model parameters: %d' % trainable_param_n)

        # 定义优化器
        optimizer = self.optim_p(self.net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
        early_stopping = EarlyStopping(patience=15, verbose=True)
        # 初始化损失记录
        trainLossList = []
        valLossList = []
        # 迭代训练max_epoch轮
        config_handler.set_global(length=25, spinner='waves2')
        with alive_bar(self.max_epoch, bar='bubbles') as bar:
            for t in range(self.max_epoch):
                train_loss = self._train(train_loader, optimizer)
                val_loss = self._val(val_loader)
                trainLossList.append(train_loss)
                valLossList.append(val_loss)
                # early_stopping(valLossList[-1]*1000, model=self.net, path=self.model_path)
                early_stopping(valLossList[-1] * 1000, model=self.net, path=self.model_path, bar=bar, label=time_label)
                if early_stopping.early_stop:
                    bar.text('Early stopping')
                    bar()
                    # print('Early stopping')
                    break
                scheduler.step()
        return valLossList


'''
定义神经网络预测模块
input: DataFrame格式的前3天历史数据和前三小时实时数据(包括液位、青东测压点压力、华翔进水压力)
output: 后五分钟预测液位值
'''


class pump_model_pred:
    def __init__(self, station_name, obj_type, selected_date=None, denose=False):
        self.denose = denose
        self.station_name = station_name
        self.obj_type = obj_type
        self.selected_date = selected_date
        self.target_name = '_'.join([self.station_name, self.obj_type])
        self.path = './{}'.format(self.target_name)
        self.scalar_path = './scalar_pump'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练采用的硬件
        self.net = self._get_model()
        self.trans = self._get_scalar()

    def _update_paras(self):
        self.net = self._get_model()
        self.trans = self._get_scalar()

    # 找到最新的模型文件
    def _get_model_weights(self):
        if self.selected_date:
            model_name = f'model_best_{self.selected_date[:4]}_{self.selected_date[4:6]}_{self.selected_date[-2:]}.pth'
            return os.path.join(self.path, model_name)
        files = os.listdir(self.path)
        if not files:
            return None
        else:
            files = sorted(files, key=lambda x: os.path.getmtime(
                os.path.join(self.path, x)))  # 格式解释:对files进行排序.x是files的元素,:后面的是排序的依据.  x只是文件名,所以要带上join.
            return os.path.join(self.path, files[-1])

    # 加载模型参数
    def _get_model(self):
        file_path = self._get_model_weights()
        net = GRU_Model(device=self.device, input_size=input_size, hidden_units_1=32, hidden_units_2=8,
                        type=self.obj_type)
        net.load_state_dict(torch.load(file_path, map_location='cpu'))
        net = net.to(self.device)
        return net

    # 加载scalar参数
    def _get_scalar(self):
        file_path = self._get_model_weights()
        filename = os.path.basename(file_path)
        scalar_name = 'scalar_' + '_'.join(os.path.splitext(filename)[0].split('_')[2:])
        scalar = joblib.load(os.path.join(self.scalar_path, scalar_name))
        return scalar

    # 提取同期特征（周期性特征）
    def _get_CycleAttr(self, data, time_index):
        # 目标特征
        cycle_fea = data[self.target_name]
        att_cycle = pd.Series()
        for i in range(1, 6):
            # 找到前i天同期的时间索引
            timelabel_di = time_index - pd.Timedelta(days=i)
            # 前后偏移2个时间步长
            time_bias = 2 * pd.Timedelta(minutes=5)
            # 找到当天同期的液位数据
            df_di = cycle_fea[timelabel_di - time_bias:timelabel_di + time_bias]
            att_cycle = att_cycle.append(df_di)
        return att_cycle.values

    # 提取前三小时最不利点压力、进水压力和时间编码特征
    def _get_DriveAttr(self, data):
        if self.obj_type == 'water_level':
            vb_column = self.station_name + '_pressure'
            if self.station_name == 'hx':
                alt_vb_column = 'xfx_pressure'
                alt_ovb_volumn = 'xfx_water_level'
            elif self.station_name == 'xfx':
                alt_vb_column = 'hx_pressure'
                alt_ovb_volumn = 'hx_water_level'
            else:
                raise ValueError

        elif self.obj_type == 'pressure':
            vb_column = self.station_name + '_water_level'
            if self.station_name == 'hx':
                alt_vb_column = 'xfx_water_level'
                alt_ovb_volumn = 'xfx_pressure'
            elif self.station_name == 'xfx':
                alt_vb_column = 'hx_water_level'
                alt_ovb_volumn = 'hx_pressure'
            else:
                raise ValueError
        # fea_columns = [vb_column, 'bottom_pressure', 'time_encode','water_demand','xj_pressure']
        fea_columns = ['bottom_pressure', 'time_encode', vb_column, alt_vb_column, 'water_demand',
                       'xj_pressure', alt_ovb_volumn]

        data = data[fea_columns]
        att_drive = data.values.T.ravel()
        return att_drive

    def _inverse_transform(self, inp, out):
        pos_dict = {'hx_water_level': 0, 'xfx_water_level': 1, 'hx_pressure': 3, 'xfx_pressure': 4}
        format_array = np.zeros((1, inp.shape[1]))
        format_array[0, pos_dict[self.target_name]] = out
        return self.trans.inverse_transform(format_array)[0, pos_dict[self.target_name]]

    def _get_treat(self, data):
        data['time_encode'] = [i.hour * 60 + i.minute for i in data.index.time]
        data[data.columns] = self.trans.transform(data.values)
        return data

    def pred(self, realtime_data, history_data):
        realtime_data = self._get_treat(realtime_data.copy())
        history_data = self._get_treat(history_data.copy())
        next_time = realtime_data.index[-1] + pd.Timedelta(minutes=5)
        att_drive = self._get_DriveAttr(realtime_data)
        att_cycle = self._get_CycleAttr(history_data, next_time)
        input = np.concatenate([att_drive, att_cycle])
        self.net.eval()  # 启动测试
        input_torch = torch.tensor(input)
        input_torch = input_torch.to(device=self.device, dtype=torch.float)
        input_torch = input_torch.view(1, -1, 1)
        out = self.net.forward(input_torch)
        out = out.squeeze()
        out = out.cpu().detach().numpy()
        output = self._inverse_transform(realtime_data, out)  # 需要按照格式调整
        return output


'''
生成调度指令
实例初始化：init_time 起始时间
调用函数：signal_cal （input:当前时刻液位，下一时刻预测液位; output:调度指令）
'''

## 用于线上代码生成所有泵站调度指令，离线模拟系统不用
# class Union_Order_Gen:
#     def __init__(self, init_time, init_status):
#         self.init_time = init_time
#         self.init_status = init_status
#         hx_yl_status = self.init_status[:3]
#         hx_yw_status = self.init_status[3:5]
#         xfx_yl_status = self.init_status[5:8]
#         xfx_yw_status = self.init_status[8:10]
#         self.hx_yw = Order_Gen('hx_water_level', self.init_time, hx_yw_status)
#         self.hx_yl = Order_Gen('hx_pressure', self.init_time, hx_yl_status)
#         self.xfx_yw = Order_Gen('xfx_water_level', self.init_time, xfx_yw_status)
#         self.xfx_yl = Order_Gen('xfx_pressure', self.init_time, xfx_yl_status)
#
#     def signal_cal(self, next_status, current_status, bottom_pressure, real_hx_yw, real_xfx_yw):
#         ## 读取当前的压力、液位数据和下一时刻预测值
#         current_hx_yw, current_xfx_yw, current_hx_yl, current_xfx_yl = current_status
#         next_hx_yw, next_xfx_yw, next_hx_yl, next_xfx_yl = next_status
#         ## 生成当前时刻的指令
#         hx_yw_signal = self.hx_yw.signal_cal(current_hx_yw, next_hx_yw)
#         xfx_yw_signal = self.xfx_yw.signal_cal(current_xfx_yw, next_xfx_yw)
#         hx_yl_signal = self.hx_yl.signal_cal(current_hx_yl, next_hx_yl)
#         xfx_yl_signal = self.xfx_yl.signal_cal(current_xfx_yl, next_xfx_yl)
#         signals = [hx_yw_signal, xfx_yw_signal, hx_yl_signal, xfx_yl_signal]
#         ## 返回指令
#         orders = []
#         if self.hx_yw.order_cal(bottom_pressure, real_hx_yw):
#             hx_tank_ord = self.hx_yw.order_cal(bottom_pressure, real_hx_yw)
#             orders.append(hx_tank_ord)
#         if self.xfx_yw.order_cal(bottom_pressure, real_xfx_yw):
#             xfx_tank_ord = self.xfx_yw.order_cal(bottom_pressure, real_xfx_yw)
#             orders.append(xfx_tank_ord)
#         if self.hx_yl.order_cal(bottom_pressure):
#             hx_pump_ord = self.hx_yl.order_cal(bottom_pressure)
#             orders.append(hx_pump_ord)
#         if self.xfx_yl.order_cal(bottom_pressure):
#             xfx_pump_ord = self.xfx_yl.order_cal(bottom_pressure)
#             orders.append(xfx_pump_ord)
#         return signals, orders


## 每个模型的指令生成
class Order_Gen:
    def __init__(self, target_name, init_time, init_status):
        # 储存上次使用的泵站信息
        self.pump_number = []
        try:
            pump_id = list(init_status).index(1)
        except:
            print('[Order Gen]:No pump open now at {}'.format(ctime()))
            pass
        else:
            if target_name == 'hx_water_level' or target_name == 'xfx_water_level':
                self.pump_number.append('#{}'.format(pump_id + 4))
            if target_name == 'hx_pressure' or target_name == 'xfx_pressure':
                self.pump_number.append('#{}'.format(pump_id + 1))
        init_cc = sum(init_status)
        if init_cc > 1:
            init_cc = 1
        # 信号序列
        self.signals = [init_cc]  # 初始化信号序列
        self.control_signals = [init_cc]  # 带控制的信号序列
        # 预测序列
        self.preds = []  # 初始化预测结果序列
        # 其他配置参数
        self.max_len = 100  # 最大序列长度
        self.time = init_time  # 起始时间
        self.flag = init_cc
        self.silence_flag = False
        self.silence_collect = []
        self.silence_counter = 0
        self.sensor_flag = 0
        self.target_name = target_name

    def _maxlen_cut(self):
        self.signals = self.signals[-self.max_len:]
        self.preds = self.preds[-self.max_len - 1:]

    ## 华翔水库液位调度逻辑
    def _hx_tank_logi(self, current_val, next_val, current_water_level):
        # 静默指令
        if self.silence_flag:
            self.silence_counter += 1
            return 0
        ## 突破阈值触发
        # 进水触发
        if (next_val - current_val > 0.025
                and next_val < 2
                and self.signals[-1] == 0):
            self.flag = -1
        # 关水触发
        if (next_val - current_val < 0.015
                and next_val > 4
                and self.signals[-1] == -1):
            self.flag = 0
        # 开泵触发
        if (next_val - current_val < -0.02
                and next_val > 3
                and self.signals[-1] == 0
                and (self.time + pd.Timedelta(minutes=5)).strftime('%H:%M') not in Interval('00:00', '06:00')):
            self.flag = 1
        # 关泵触发
        if (-0.025 < next_val - current_val < 0
                and next_val < 2
                and self.signals[-1] == 1):
            self.flag = 0
        ## 强制触发逻辑
        # 强制液位触发
        if current_water_level < 1.1 and self.flag == 1:
            self.flag = 0
        if (current_water_level < 1.1
                and self.flag == 0
                and (self.time + pd.Timedelta(minutes=5)).strftime('%H:%M') in Interval('06:00', '18:00')):
            self.flag = -1
        # 强制时段触发
        if ((self.time + pd.Timedelta(minutes=5)).strftime('%H:%M') in Interval('00:00', '06:00')
                and self.flag == 1):
            self.flag = 0
        if ((self.time + pd.Timedelta(minutes=5)).strftime('%H:%M') in Interval('13:00', '16:00')
                and self.flag == 1):
            self.flag = 0
        if ((self.time + pd.Timedelta(minutes=5)).strftime('%H:%M') in Interval('01:00', '02:00')
                and self.flag == 0):
            self.flag = -1

    ## 新凤溪水库液位调度逻辑
    def _xfx_tank_logi(self, current_val, next_val, current_water_level):
        ## 静默指令
        if self.silence_flag:
            self.silence_counter += 1
            return 0
        ## 突破阈值触发
        # 进水触发
        if ((next_val - current_val) > 0.025
                and next_val < 2
                and ((self.time + pd.Timedelta(minutes=5)).strftime('%H:%M') in Interval('00:00', '02:00')
                     or (self.time + pd.Timedelta(minutes=5)).strftime('%H:%M') in Interval('23:00', '23:55'))
                and self.signals[-1] == 0):
            self.flag = -1
        # 关水触发
        if (next_val - current_val < 0.015
                and next_val > 3
                and self.signals[-1] == -1):
            self.flag = 0
        # 开泵触发
        if (((next_val - current_val) < -0.015
             and next_val > 2
             or self.time.strftime('%H:%M') in Interval('18:00', '21:00'))
                and self.signals[-1] == 0):
            self.flag = 1

        # 关泵触发
        if (((abs(next_val - current_val) < 0.003 and next_val < 3) or next_val < 1.8)
                and self.signals[-1] == 1
                and self.time.strftime('%H:%M') not in Interval('18:00', '21:00')):
            self.flag = 0

        ## 强制触发逻辑
        # 强制液位触发
        if (current_water_level < 1.1
                and self.flag == 1):
            self.flag = 0
        if (current_water_level < 1.1
                and self.flag == 0
                and (self.time + pd.Timedelta(minutes=5)).strftime('%H:%M') in Interval('06:00', '18:00')):
            self.flag = -1

        # 强制时段触发
        if ((self.time + pd.Timedelta(minutes=5)).strftime('%H:%M') in Interval('00:00', '06:00')
                and self.flag == 1):
            self.flag = 0
        if ((self.time + pd.Timedelta(minutes=5)).strftime('%H:%M') in Interval('13:00', '16:00')
                and self.flag == 1):
            self.flag = 0
        if ((self.time + pd.Timedelta(minutes=5)).strftime('%H:%M') in Interval('01:00', '02:00')
                and self.flag == 0):
            self.flag = -1

    ## 华翔增压泵指令的逻辑
    def _hx_pressure_logi(self, current_val, next_val, current_pressure):
        # 静默指令
        if self.silence_flag:
            self.silence_counter += 1
            return 0
        ## 阈值触发逻辑
        # 开泵触发
        if (next_val > 198
                and next_val - current_val > 3.5
                and self.signals[-1] == 0
                and self.time.strftime("%H:%M") not in Interval('00:00', '05:00')):
            self.flag = 1
        # 关泵触发
        if (next_val < 190
                and ((abs(next_val - current_val) < 0.5) or (next_val < 210 and next_val - current_val < -2))
                and self.signals[-1] == 1):
            self.flag = 0

        if (current_pressure < 165
                and self.signals[-1] == 0
                and self.time.strftime("%H:%M") in Interval('05:05', '09:00')):
            self.sensor_flag = 1
        if (self.time.strftime("%H:%M") in Interval('00:00', '05:00')
                or self.time.strftime("%H:%M") in Interval('23:00', '23:55')):
            self.flag = 0
            if self.sensor_flag == 1:
                self.sensor_flag = 0

    ## 新凤溪增压泵指令的逻辑
    def _xfx_pressure_logi(self, current_val, next_val, current_pressure):
        # 静默模式
        if self.silence_flag:
            self.silence_counter += 1
            return 0
        if (next_val > 208
                and abs(next_val - current_val) > 2
                and self.signals[-1] == 0
                and self.time.strftime("%H:%M") not in Interval('00:00', '05:00')):
            self.flag = 1
        if (next_val < 200
                and abs(next_val - current_val) < 2
                and self.signals[-1] == 1):
            self.flag = 0

        if (current_pressure < 171
                and self.signals[-1] == 0
                and self.time.strftime("%H:%M") in Interval('05:05', '09:00')):
            self.sensor_flag = 1
        if (self.time.strftime("%H:%M") in Interval('00:00', '05:00')
                or self.time.strftime("%H:%M") in Interval('23:00', '23:55')):
            self.flag = 0
            if self.sensor_flag == 1:
                self.sensor_flag = 0

    ## 选择开哪台水库泵（华翔/新凤溪）
    def _get_tank_number(self):
        all_pumps = ['#4', '#5']
        if len(self.pump_number):
            all_pumps.remove(self.pump_number[-1])
            use_engine = all_pumps[0]
        else:
            use_engine = np.random.choice(all_pumps)
        self.pump_number.append(use_engine)
        return use_engine

    ## 选择开哪台增压泵（华翔/新凤溪）
    def _get_pump_number(self):
        all_pumps = ['#1', '#2', '#3']
        if len(self.pump_number):
            all_pumps.remove(self.pump_number[-1])
            use_engine = np.random.choice(all_pumps)
        else:
            use_engine = np.random.choice(all_pumps)
        self.pump_number.append(use_engine)
        return use_engine

    # 计算下一时间步的指令信号
    def signal_cal(self, current_val, pred_val, control_value):

        self.time += pd.Timedelta(minutes=5)
        if len(self.preds) != 0:
            current_val = self.preds[-1]
            # print('上一时刻液位：{}'.format(current_yw))
            # print(pred_yw - current_yw, pred_yw, self.signals[-1])
            if self.target_name == 'hx_water_level':
                self._hx_tank_logi(current_val, pred_val, control_value)
            if self.target_name == 'xfx_water_level':
                self._xfx_tank_logi(current_val, pred_val, control_value)
            if self.target_name == 'hx_pressure':
                self._hx_pressure_logi(current_val, pred_val, control_value)
            if self.target_name == 'xfx_pressure':
                self._xfx_pressure_logi(current_val, pred_val, control_value)
            ## 静默指令计时
            if self.silence_counter == 8:
                self.silence_flag = False
                self.silence_counter = 0
                self.silence_collect = []
            if self.silence_flag:
                self.silence_collect.append(self.flag)
            self.signals.append(self.flag)
            if self.sensor_flag == 0:
                self.control_signals.append(self.flag)
            elif self.sensor_flag == 1:
                if self.flag == 0:
                    self.control_signals.append(1)
                elif self.flag == 1:
                    self.control_signals.append(self.flag)
                    self.sensor_flag = 0
                else:
                    print('invalid sensor flag')



        else:
            self.signals.append(self.signals[-1])
            self.control_signals.append(self.control_signals[-1])

        ## 测试静默状态用
        # print('静默模式状态为：{}'.format(self.silence_flag))
        # print('静默计数器：{}'.format(self.silence_counter))
        # print('静默数组：{}'.format(self.silence_collect))
        # append 预测值
        # if self.sensor_flag == 1:
        #     print(self.time, 'flag', self.flag, 'sensor_flag', self.sensor_flag)
        #     print(self.signals[-5:])
        #     print(self.control_signals[-5:])

        self.preds.append(pred_val)
        self._maxlen_cut()
        return self.signals[-1]

    # 生成指令内容
    def order_cal(self, *reasons_data):
        message = None
        current_pressure, current_water_level = reasons_data
        status = self.control_signals[-1]
        use_engine = self._get_tank_number()
        if self.control_signals[-1] != self.control_signals[-2]:
            if self.control_signals[-2] == 0 and self.control_signals[-1] == -1:
                if self.target_name == 'hx_water_level':
                    ## 指令内容
                    with open('hx_duration_model.pkl', 'rb') as f:
                        hx_duration = pickle.load(f)
                    time_inp = get_input((self.time + pd.Timedelta(minutes=5)))
                    duration_time = hx_duration.solve(time_inp)
                    message = hx_duration_order(self.time, duration_time, current_water_level).__dict__
                if self.target_name == 'xfx_water_level':
                    ## 指令内容
                    with open('xfx_duration_model.pkl', 'rb') as f:
                        xfx_duration = pickle.load(f)
                    time_inp = get_input((self.time + pd.Timedelta(minutes=5)))
                    duration_time = xfx_duration.solve(time_inp)
                    message = xfx_duration_order(self.time, duration_time, current_water_level).__dict__
            if self.control_signals[-2] == 0 and self.control_signals[-1] == 1:
                if self.target_name == 'hx_water_level':
                    ## 指令内容
                    message = hx_pump_order(self.target_name, self.time, status,
                                            use_engine, current_pressure, current_water_level).__dict__
                if self.target_name == 'xfx_water_level':
                    ## 指令内容
                    message = xfx_pump_order(self.target_name, self.time, status,
                                            use_engine, current_pressure, current_water_level).__dict__
                if self.target_name == 'hx_pressure':
                    ## 指令内容
                    message = hx_pump_order(self.target_name, self.time, status,
                                             use_engine, current_pressure, current_water_level).__dict__
                if self.target_name == 'xfx_pressure':
                    ## 指令内容
                    message = xfx_pump_order(self.target_name, self.time, status,
                                            use_engine, current_pressure, current_water_level).__dict__
            if self.control_signals[-2] == 1 and self.control_signals[-1] == 0:
                if self.target_name == 'hx_water_level':
                    ## 指令内容
                    message = hx_pump_order(self.target_name, self.time, status,
                                            use_engine, current_pressure, current_water_level).__dict__
                if self.target_name == 'xfx_water_level':
                    ## 指令内容
                    message = xfx_pump_order(self.target_name, self.time, status,
                                            use_engine, current_pressure, current_water_level).__dict__
                if self.target_name == 'hx_pressure':
                    ## 指令内容
                    message = hx_pump_order(self.target_name, self.time, status,
                                             use_engine, current_pressure, current_water_level).__dict__
                if self.target_name == 'xfx_pressure':
                    ## 指令内容
                    message = xfx_pump_order(self.target_name, self.time, status,
                                             use_engine, current_pressure, current_water_level).__dict__

        return message


