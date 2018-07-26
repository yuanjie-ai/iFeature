# -*- coding: utf-8 -*-
"""
__title__ = '_TimeFeature'
__author__ = 'JieYuan'
__mtime__ = '2018/7/22'
"""


class TimeFeature(object):
    def __init__(self):
        pass

    @staticmethod
    def get_feats_time(data, group_col=None, time_col='CREATETIME', feat_cols=None):
        """
        :param data:
        :param group_col:
        :param time_col:
        :param feat_cols:
        :return:
        """
        print('time continuous ...')
        data[time_col + '_year'] = data[time_col].apply(lambda x: x.year)
        data[time_col + '_month'] = data[time_col].apply(lambda x: x.month)
        data[time_col + '_day'] = data[time_col].apply(lambda x: x.day)
        data[time_col + '_weekday'] = data[time_col].apply(lambda x: x.weekday())
        data[time_col + '_diff'] = data.groupby(group_col)[time_col].diff().apply(lambda x: x.days).fillna(0)
        data[time_col + '_time_interval'] = data[time_col].transform(lambda x: x.max() - x).apply(lambda x: x.days)

        if feat_cols:
            print("Feats Diff ...")
            _data = data.groupby(group_col)[feat_cols].diff().fillna(0) \
                .rename(columns={i + '_diff': i for i in feat_cols})  # 提前按时间升序排列

            print("Feats Average Encoding ...")
            _data = data.groupby(time_col)[feat_cols].transform('mean') \
                .rename(columns={i + '_diff': i for i in feat_cols})  # median
        return data
