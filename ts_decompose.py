import pywt as wt
import pandas as pd
import matplotlib.pyplot as plt

class wavelet_strcuture(object):
    def __init__(self, ts_array, wave_func='db4'):
        self.ts_array = ts_array.copy()
        self.wave_func = wave_func
        self.cA3 = self._ts_decompose()[0]
        self.cD3 = self._ts_decompose()[1]
        self.cD2 = self._ts_decompose()[2]
        self.cD1 = self._ts_decompose()[3]
    def _ts_decompose(self):
        cA3, cD3, cD2, cD1 = wt.wavedec(self.ts_array, self.wave_func, level=3, mode='periodic')
        cA3_array = wt.waverec([cA3, None, None, None], self.wave_func)[:len(self.ts_array)]
        cD3_array = wt.waverec([None, cD3, None, None], self.wave_func)[:len(self.ts_array)]
        cD2_array = wt.waverec([None, cD2, None], self.wave_func)[:len(self.ts_array)]
        cD1_array = wt.waverec([None, cD1], self.wave_func)[:len(self.ts_array)]
        return cA3_array, cD3_array, cD2_array, cD1_array

def wavelet_denosing(df,mode='all'):
    data = df.copy()
    selected_columns = data.columns if mode == 'all' else mode
    for cl in selected_columns:
        cl_array = data[cl].values
        cl_array_cA3 = wavelet_decompose(cl_array)[0]
        data.loc[:, cl] = cl_array_cA3
    return data





def wavelet_decompose(data, wave_func='db4'):
    Wave_func = wt.Wavelet(wave_func)
    cA3, cD3, cD2, cD1 = wt.wavedec(data, Wave_func, level=3, mode='periodic')
    cA3_array = wt.waverec([cA3, None, None, None], Wave_func)[:len(data)]
    cD3_array = wt.waverec([None, cD3, None, None], Wave_func)[:len(data)]
    cD2_array = wt.waverec([None, cD2, None], Wave_func)[:len(data)]
    cD1_array = wt.waverec([None, cD1], Wave_func)[:len(data)]
    return cA3_array,cD3_array, cD2_array, cD1_array

def wavelet_sample_construct(data, wave_func='db4'):
    ts = data['value'].values
    mkdf = lambda x: pd.DataFrame(x, columns=['value'], index=data.index)
    cA3, cD3, cD2, cD1 = list(map(mkdf, wavelet_decompose(ts, wave_func)))
    return cA3, cD3, cD2, cD1


if __name__ == '__main__':
    train_data = pd.read_csv('./train_dataset.csv', index_col='datetime', parse_dates=True)
    train_data.dropna(inplace=True)
    cA3_data = wavelet_denosing(train_data[['hx_pressure']])
    plt.plot(train_data.index, cA3_data['hx_pressure'])
    plt.plot(train_data.index, train_data['hx_pressure'])
    plt.show()
    # df = STData(data_path, precision=15, sp=0)
    # # 过滤异常值
    # df.Z_score_Filter(lag=10000, threshold=3, plot=False)
    # # 异常值插补
    # df.TW_KNNR_imputation()
    # dataclip = df.data['2020-9':'2020-11']  # represent the data of september to November
    # cA3, cD3, cD2, cD1 = wavelet_sample_construct(dataclip)

