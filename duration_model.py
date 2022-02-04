import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import scipy.stats as st
import joblib
import pickle
'''
定义全局变量
'''
# df_hx = pd.read_csv('./data/hx_js.csv', header=0)
# df_xfx = pd.read_csv('./data/xfx_js.csv', header=0)

def get_time_tuple(df):
    res = []
    get_operate_date = lambda x:(pd.to_datetime(x) + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')
    get_operator_time = lambda x:pd.to_datetime(x)
    df['date'] = df['指令时间'].apply(get_operate_date)
    df['time'] = df['指令时间'].apply(get_operator_time)
    for date, item in df.groupby('date'):
        ## 搜索进水指令
        cond_start = item['指令'] == '进水'
        if sum(cond_start) == 1:
            start_time = item[cond_start].iloc[0,-1]
        elif sum(cond_start) == 0:
            start_time = 0
            print(f'{date} 没有进水指令')
        else:
            start_time = 0
            print(f'{date} 多条进水指令')
        ## 搜索关水指令
        cond_end = item['指令'] == '关水'
        if sum(cond_end) == 1:
            end_time = item[cond_end].iloc[0, -1]
        elif sum(cond_end) == 0:
            end_time = 0
            print(f'{date} 没有关水指令')
        else:
            end_time = 0
            print(f'{date} 多条关水指令')
        if start_time!= 0 and end_time != 0:
            duration = (end_time-start_time).seconds/3600
            res_record = [date, start_time, end_time, duration]
            res.append(res_record)
    df_res = pd.DataFrame(res, columns=['date','start_time','end_time','duration'])
    return df_res

def get_std_time(x):
    hour = int(x.strftime('%H'))
    if hour > 21:
        return pd.to_datetime(x.strftime('%H:%M:%S'))- pd.Timedelta(days=1)
    else:
        return pd.to_datetime(x.strftime('%H:%M:%S'))

def get_dig_time(x):
    init_pos = pd.to_datetime('22:00:00')
    return (x-init_pos).seconds/60

def get_input(t):
    return get_dig_time(get_std_time(t))

class linear_reg(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def fit(self):
        slope, intercept,r_value, p_value,std_err = st.linregress(self.x, self.y)
        self.slope = slope
        self.intercept = intercept
        return slope, intercept, r_value**2
    def solve(self,x): # numpy array
        return self.slope*x+self.intercept


if __name__ == '__main__':
    df_hx_res = get_time_tuple(df_hx)
    index = df_hx_res['start_time'].apply(get_std_time)
    dig_index = index.apply(get_dig_time)
    labels = index.apply(lambda x: x.strftime('%H:%M:%S'))
    digital_index = df_hx_res['start_time'].apply(get_dig_time)
    rg_model = linear_reg(dig_index.values, df_hx_res['duration'].values)
    # joblib.dump(rg_model, 'xfx_duration_model')



    slope, intercept, r_square = rg_model.fit()
    with open('hx_duration_model.pkl','wb') as f:
        pickle.dump(rg_model, f)
    x_lin = np.linspace(0, 210,1000)
    y_lin = rg_model.solve(x_lin)
    plt.plot(x_lin, y_lin, color='indianred',ls='--')
    plt.scatter(digital_index, df_hx_res['duration'])
    plt.text(25,9,'Regression Equation\n y= {slope:.3f}x+{intercept:.2f}\n $R^2={r2:.2f}$'.format(slope=slope,
                                                                        intercept=intercept, r2=r_square))
    # dig_index.drop_duplicates(inplace=True)
    # labels.drop_duplicates(inplace=True)
    x = range(0,210,10)
    x_label = [(pd.to_datetime('22:00') + pd.Timedelta(minutes=xi)).strftime('%H:%M') for xi in x]
    plt.xticks(x, x_label)
    plt.xlim(20, 190)
    plt.ylim(3, 10)
    plt.show()
    print(df_hx_res)