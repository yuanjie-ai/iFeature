# -*- coding: utf-8 -*-
"""
__title__ = '_TimeFeature'
__author__ = 'JieYuan'
__mtime__ = '2018/7/22'
"""
from tqdm import tqdm


class TimeFeature(object):
    def __init__(self):
        pass

    @staticmethod
    def get_feats_time(data, group=None, feats=['numerical', ], ts='ts'):
        """时间的聚合特征同数值型
        与时间相关特征的特征衍生的非聚合特征
        :param data:
        :param group: "id"
        :param feats:
        :param ts:
        :return:
        """
        print('time continuous ...')
        data['ts_year'] = data[ts].apply(lambda x: x.year)
        data['ts_month'] = data[ts].apply(lambda x: x.month)
        data['ts_day'] = data[ts].apply(lambda x: x.day)
        data['ts_weekday'] = data[ts].apply(lambda x: x.weekday())
        data['ts_diff'] = data.groupby(group)[ts].diff().apply(lambda x: x.days).fillna(0)  ##########
        data['ts_time_interval'] = data[ts].agg(lambda x: x.max() - x).apply(lambda x: x.days)
        if feats:  # 对时间特征可用数值特征平均编码
            print("ts_average_encoding ...")
            gr = data.groupby(ts)
            for i in tqdm(feats):
                data['ts_average_encoding_' + i] = gr[i].transform('mean')  # median

            print("feats diff ...")
            gr = data.groupby(group)
            for i in tqdm(feats):  # 数值特征也可以按时间顺序进行差分
                data['diff_' + i] = gr[i].diff().fillna(0)
        return data
