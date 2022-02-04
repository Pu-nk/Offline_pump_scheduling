import pandas as pd
import random
from pandas import DatetimeIndex

from offline_class_package import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
a = 3
## 全局参数配置
station_name = 'xfx'  # 泵站名称：新凤溪
obj_type = 'water_level'  # 预测目标: 液位

## 生成target_name
target_name = '_'.join([station_name, obj_type])
init_status = [0, 0]

need_order = True  # 是否生成指令
plot = True  # 是否画图
res_columns = ['obs_val', 'pred_val', 'signal']  # 结果文件的表头

## 设置预测参数
start_time, end_time = pd.to_datetime('2022/1/19 00:00:00'), pd.to_datetime('2022/1/20 23:55:00')  # 模拟时段设置
# df = pd.read_csv('./data/historydata.csv', header=0, index_col='datetime', parse_dates=True)
df = pd.read_csv('./data/train_data_2022_01_21_12.csv', header=0, index_col='datetime', parse_dates=True)  # 需要模拟的数据文件
order_Generator = Order_Gen(target_name, start_time, init_status)  # 初始化指令生成器


## 手动设置训练数据
def get_selected_data():
    # train_data = pd.read_csv('./data/2020_9_train_data.csv', index_col='datetime', parse_dates=True)
    train_data = pd.read_csv('./data/train_dataset.csv', index_col='datetime', parse_dates=True)
    train_data.dropna(inplace=True)
    # train_data = train_data.resample('5min').first()['2020-9']
    return train_data


## 设置泵站状态数据
# power_queue = pd.read_csv('./power_data_2022_01_20_10.csv', header=0, index_col='Unnamed: 0', parse_dates=True)
# def get_RPS():
#     rt_pump_status = power_queue.get()
#     hx_pump_status = 1 if sum(rt_pump_status.iloc[-1, -10:-7]) > 0 else 0
#     hx_tank_status = 1 if sum(rt_pump_status.iloc[-1, -7:-5]) > 0 else 0
#     xfx_pump_status = 1 if sum(rt_pump_status.iloc[-1, -5:-2]) > 0 else 0
#     xfx_tank_status = 1 if sum(rt_pump_status.iloc[-1, -2:]) > 0 else 0
#     return {'HXYL':hx_pump_status,
#             'HXYW':hx_tank_status,
#             'XFXYL':xfx_pump_status,
#             'XFXYW': xfx_tank_status}


def setup_seed(seed):  # 设置随机数种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_data(file_path='./pump_train_data'):  # 自动从数据文件夹下获取训练数据
    files = os.listdir(file_path)
    if not files:
        print("[offline Train thread] Cannot Find Train Data:", ctime())
        return None
    else:
        files = sorted(files, key=lambda x: os.path.getmtime(
            os.path.join(file_path, x)))  # 格式解释:对files进行排序.x是files的元素,:后面的是排序的依据.  x只是文件名,所以要带上join.
        return os.path.join(file_path, files[-1])


def model_train(auto_data=True, data_api=None):  # 模型训练
    """
     :param auto_data: 是否自动获取训练数据，默认为True
     :param data_api: 若auto_data = False, 则指定手动数据接口
     :return None
    """
    print("[offline Train thread] Start Model Training at:", ctime())
    # 获取history数据
    if auto_data:
        train_data = pd.read_csv(get_train_data(), index_col='Unnamed: 0', parse_dates=True)
        print('Auto Extract Recently Train Data\n', train_data)
    else:
        train_data = data_api()
        print('Self-Defined Train Data\n', train_data)

    model_train_machine = pump_model_train(station_name, obj_type)
    model_final_loss = model_train_machine.epoches_train(train_data)
    print(model_final_loss[-1] * 1000)
    print("[offline Train thread] Finish {} {} Model Training at:".format(station_name, obj_type), ctime())


def model_pred(auto_model=True, model_date=None):  # 模型预测
    """
    :param auto_model: 是否自动获取最新的模型，默认为True
    :param model_date: 如果auto_model = False, 手动指定对应日期的模型
    :return:
    """
    if auto_model:
        model_pred_machine = pump_model_pred(station_name, obj_type)
    else:
        model_pred_machine = pump_model_pred(station_name, obj_type, model_date)
    ## 生成采样日期，每5min一个
    sample_dates = pd.date_range(start=start_time, end=end_time, freq='5min')
    outputs = []
    ## 预测结果表
    result = pd.DataFrame(columns=res_columns)
    for date in sample_dates:
        end_dates = date
        start_dates = date - pd.Timedelta(minutes=175)  # 前三小时
        # 提取实时数据
        realtime_data = df[start_dates:end_dates]  # 前三小时的实时数据
        # 提取前7天数据
        seven_end_date = date - pd.Timedelta(days=1) + pd.Timedelta(minutes=15)
        seven_start_date = seven_end_date - pd.Timedelta(days=7)
        history_data = df[seven_start_date:seven_end_date]
        history_data = history_data[1:]
        # print(history_data)
        # 输出预测结果
        next_val = model_pred_machine.pred(realtime_data, history_data)
        if need_order:
            ## 计算指令和状态信号
            if (target_name == 'hx_water_level') or (target_name == 'xfx_water_level'):
                current_val = realtime_data[target_name].iloc[-1]
                current_water_level = current_val
                bottom_pressure = realtime_data['bottom_pressure'].iloc[-1]
                # print(date)
                signal = order_Generator.signal_cal(current_val, next_val, current_water_level)
                res = pd.DataFrame([[current_val, next_val, signal]], columns=res_columns, index=[date])
                result = result.append(res)
                if order_Generator.order_cal(bottom_pressure, current_water_level):
                    order = order_Generator.order_cal(bottom_pressure, current_water_level)
                    print(order)
                    ## 测试指令拒绝与接受静默机制
                    # Response = eval(input('请输入回复指令(接受-2 拒绝-4):'))
                    # Response = 2
                    # if Response == 2:
                    #     order_Generator.silence_flag = True
                    # elif Response == 4:
                    #     print('当前信号序列末置位为：{}'.format(order_Generator.control_signals[-1]))
                    #     order_Generator.control_signals[-1] = order_Generator.control_signals[-2]
                    #     print('当前信号序列末置位为：{}'.format(order_Generator.control_signals[-1]))
                    #     order_Generator.signals[-1] = order_Generator.signals[-2]
                    #     print('当前泵站状态为：{}'.format(order_Generator.flag))
                    #     order_Generator.flag = order_Generator.signals[-1]
                    #     print('当前泵站状态为：{}'.format(order_Generator.flag))
                    #     order_Generator.silence_flag = True
                    #     print('静默模式状态为：{}'.format(order_Generator.silence_flag))
                    ##  写入指令
                    # with open('./record.txt', 'a') as f:
                    #     f.write(str(order) + '\n')

            if (target_name == 'hx_pressure') or (target_name == 'xfx_pressure'):
                current_val = realtime_data[target_name].iloc[-1]
                bottom_pressure = realtime_data['bottom_pressure'].iloc[-1]
                signal = order_Generator.signal_cal(current_val, next_val, bottom_pressure)
                # print(signal, date)
                res = pd.DataFrame([[current_val, next_val, signal]], columns=res_columns, index=[date])
                result = result.append(res)
                if order_Generator.order_cal(bottom_pressure):
                    order = order_Generator.order_cal(bottom_pressure)
                    print(order)
        outputs.append(next_val)
    result.to_csv('./res/{tn}_result.csv'.format(tn=target_name))  # 储存对应的预测结果
    if plot:
        label_dict = {'hx_water_level': '华翔泵站液位',
                      'hx_pressure': '华翔泵站压力',
                      'xfx_water_level': '新凤溪泵站液位',
                      'xfx_pressure': '新凤溪泵站压力'}
        fig, ax = plt.subplots(3, 1, figsize=(16, 7), sharex=True)
        obs = df[target_name][start_time + pd.Timedelta(minutes=5):end_time + pd.Timedelta(minutes=5)].values
        idx = pd.date_range(start=start_time + pd.Timedelta(minutes=5), end=end_time + pd.Timedelta(minutes=5),
                            freq='5min')
        diff = np.diff(outputs, prepend=0)
        diff[0] = 0
        ax[0].plot(idx, obs, c='gray', label='obs')
        ax[0].plot(idx, outputs, c='blue', label='pred')
        ax[0].set_ylabel(label_dict[target_name], fontsize=12)
        ax[1].plot(idx, diff, c='blue', label='diff')
        ax[1].set_ylabel('差分值', fontsize=12)
        ax[2].plot(result.index, result['signal'])
        ax[2].set_ylabel('泵站信号', fontsize=12)
        ax[0].legend(loc='upper right', fontsize=12)
        ax[1].legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        plt.savefig('./fig/{tn}_pred_plot'.format(tn=target_name)) ## 储存图片


if __name__ == '__main__':
    ## 指定随机数种子
    # setup_seed(1)
    ## 指定数据集训练
    # model_train(auto_data=False, data_api=get_selected_data)
    ## 预测
    model_pred()
    ## 指定模型预测
    # model_pred(auto_model=False, model_date='20211003')


