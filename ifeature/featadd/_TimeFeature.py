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
    def get_feats_time(df, group_col=None, time_col='CREATETIME', feat_cols=None):
        """
        :param df:
        :param group_col:
        :param time_col:
        :param feat_cols:
        :return:
        """
        print('Time Explodes Into Year/Month/Day ...')
        df[time_col + '_year'] = df[time_col].apply(lambda x: x.year)
        df[time_col + '_month'] = df[time_col].apply(lambda x: x.month)
        df[time_col + '_day'] = df[time_col].apply(lambda x: x.day)
        df[time_col + '_weekday'] = df[time_col].apply(lambda x: x.weekday())
        df[time_col + '_diff'] = df.groupby(group_col)[time_col].diff().apply(lambda x: x.days).fillna(0)
        df[time_col + '_time_interval'] = df[time_col].transform(lambda x: x.max() - x).apply(lambda x: x.days)

        if feat_cols:
            print("Feats Diff ...")
            df = df.groupby(group_col)[feat_cols].diff().fillna(0) \
                .rename(columns={i + '_diff': i for i in feat_cols})  # 提前按时间升序排列

            print("Feats Average Encoding ...")
            df = df.groupby(time_col)[feat_cols].transform('mean') \
                .rename(columns={i + '_diff': i for i in feat_cols})  # median
        return df
