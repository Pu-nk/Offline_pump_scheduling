## Offline pump scheduling system

------

An offline pump scheduling simulation system 

## Getting Started

----

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Your environment needs the following python package, you can use `conda` or `pip` to install ：

```bash
alive_progress==2.0.0
interval==1.0.0
joblib==1.0.1
matplotlib==3.3.4
numpy==1.20.2
pandas==1.2.4
PyWavelets==1.1.1
scikit_learn==1.0.2
scipy==1.6.2
torch==1.8.0
```

### File Structure

An introduction of the usage for each file:

**Filefolder**

`data` : some data for train or test

`rg` :  some static parameters

`scalar_pump`:  the scalar file for data normalization 

`res`: the `csv` file for prediction result

`fig`: the figure that visualize prediction results

`hx_water_level`, `xfx_water_level`, `hx_pressure`, `xfx_pressure`: the corresponding models for different  object

**File**

`duration_model.py`: implementation of pump duration time prediction

`offline_class_package.py`: the ralated class definition in offline pump scheduling system

`offline_test_frame.py`: main program of offline  pump scheduling system

## Running the tests

The following instructions will give you some notes about how to excute the program:

1. open `offline_test_frame.py`

2. Select  the object you want to predict, for example:

   ```python
   ## 全局参数配置
   station_name = 'xfx'  # 泵站名称：新凤溪
   obj_type = 'water_level'  # 预测目标: 液位
   ```
3. Then, if you want to train a new model, you need to select the train dataset. Two ways to configure the train dataset are implemented in the system:

   - Automatic selection of the latest train dataset, most cases are used to automatically update the model

     ```python
     model_train(auto_data=True, data_api=None)
     ```

   - Manual selection of training dataset (**Recommend**)	

     ```python
     def get_selected_data():
         # train_data = pd.read_csv('./data/2020_9_train_data.csv', index_col='datetime', parse_dates=True)
         train_data = pd.read_csv('./data/train_dataset.csv', index_col='datetime', parse_dates=True)
         train_data.dropna(inplace=True)
         # train_data = train_data.resample('5min').first()['2020-9']
         return train_data
     model_train(auto_data=False, data_api=get_selected_data)
     ```

 4.   config the `start_time` and `end_time` to determine the simulation time period and load the test dataset. It is noted that the `start_time` and `end_time` must contain in the test dataset.

      ```python
      start_time, end_time = pd.to_datetime('2022/1/19 00:00:00'), pd.to_datetime('2022/1/20 23:55:00')  # 模拟时段设置
      df = pd.read_csv('./data/train_data_2022_01_21_12.csv', header=0, index_col='datetime', parse_dates=True)
      ```

	5.  Select the random seed and predict the pump scheduling order during the simulation time period, you can also specify the model to pass the parameters `model_date`

     ```python
     ## 指定随机数种子
     setup_seed(1)
     ## 预测
     model_pred()
     ## 指定模型预测
     # model_pred(auto_model=False, model_date='20211003')
     ```

     

## Authors

- **Zhengheng Pu** - *Initial work* - [Pu-nk](https://github.com/Pu-nk)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://gist.github.com/PurpleBooth/LICENSE.md) file for details