import pandas as pd


class hx_duration_order:
    def __init__(self, time, duration_time, current_water_leve):
        self.flag = 5
        self.uuid = 'HXYW{}'.format(time.strftime('%Y%m%d%H%M'))
        self.hx_order = 0
        self.vavleId = 21640
        self.duration = round(duration_time, 1)
        self.content = '华翔水库泵站需进水'
        self.reasons = dict()
        self.reasons['390'] = current_water_leve
        self.reasons['specific_reasons'] = '夜间低峰时段开始蓄水'
        self.excute_time = (time + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")

class xfx_duration_order:
    def __init__(self, time, duration_time, current_water_leve):
        self.flag = 5
        self.uuid = 'XFXYW{}'.format(time.strftime('%Y%m%d%H%M'))
        self.xfx_order = 0
        self.vavleId = 21638
        self.duration = round(duration_time, 1)
        self.content = '新凤溪水库泵站需进水'
        self.reasons = dict()
        self.reasons['9411'] = current_water_leve
        self.reasons['specific_reasons'] = '夜间低峰时段开始蓄水'
        self.excute_time = (time + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")

class hx_pump_order:
    def __init__(self, target_name, time, status, use_engine,
                 current_pressure, current_water_level):
        self.flag = 4
        if target_name == 'hx_water_level':
            self.uuid = 'HXYW{}'.format(time.strftime('%Y%m%d%H%M'))
        if target_name == 'hx_pressure':
            self.uuid = 'HXYL{}'.format(time.strftime('%Y%m%d%H%M'))

        self.hx_order = 0
        self.pumpId = 50
        self.pumpStatus = status
        self.context = '华翔水库泵站开{}泵'.format(use_engine)
        self.reasons = dict()
        if target_name == 'hx_water_level':
            self.reasons['7015'] = current_pressure  # 这个值需要传进去
            self.reasons['390'] = current_water_level  # 水库液位
            self.specific_reasons = '用水高峰期，最不利点压力低，水库液位高'
        if target_name == 'hx_pressure':
            self.reasons['7015'] = current_pressure  # 这个值需要传进去
            self.specific_reasons = '用水高峰期，最不利点压力低'

        self.excute_time = (time + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")


class xfx_pump_order:
    def __init__(self, target_name, time, status, use_engine,
                 current_pressure, current_water_level):
        self.flag = 4
        if target_name == 'xfx_water_level':
            self.uuid = 'XFXYW{}'.format(time.strftime('%Y%m%d%H%M'))
        if target_name == 'xfx_pressure':
            self.uuid = 'XFXYL{}'.format(time.strftime('%Y%m%d%H%M'))
        self.xfx_order = 0
        self.pumpId = 696
        self.pumpStatus = status
        self.context = '新凤溪水库泵站开{}泵'.format(use_engine)
        self.reasons = dict()
        if target_name == 'xfx_water_level':
            self.reasons['7015'] = current_pressure  # 这个值需要传进去
            self.reasons['9411'] = current_water_level  # 水库液位
            self.specific_reasons = '用水高峰期，最不利点压力低，水库液位高'
        if target_name == 'xfx_pressure':
            self.reasons['7015'] = current_pressure  # 这个值需要传进去
            self.specific_reasons = '用水高峰期，最不利点压力低'
        self.excute_time = (time + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")


