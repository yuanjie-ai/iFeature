# -*- coding: utf-8 -*-
"""
__title__ = 'MissingFeature'
__author__ = 'JieYuan'
__mtime__ = '2018/9/29'
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ..sub import FeatureFilter


class MissingFeature(object):
    def __init__(self, df):
        self.df = df
        self.df_is_null_unique = self.__drop_duplicates()  # df.isnull().T.drop_duplicates().T完全相同的缺失值特征干掉
        self.na_group = self.__na_group()

    def __drop_duplicates(self):
        print('Drop duplicates ...')
        p = self.df.isnull()
        s = set()
        _len = len(p.columns)
        for i in range(_len - 1):
            if i in s:
                continue
            for j in range(i + 1, _len):
                if j in s:
                    continue
                elif p[p.columns[i]].tolist() == p[p.columns[j]].tolist():
                    s.add(j)
        feats = set(range(_len)) - s
        return p[p.columns[list(feats)]]

    def __na_group(self):
        # 按照相关性讲缺失特征分组
        print('Grouping ...')
        ff = FeatureFilter(self.df_is_null_unique)
        ff.filter_correlation(threshold=0.9)
        na_group = ff.corr_record[lambda x: x.corr_feature.duplicated(keep=False)] \
            .groupby('corr_feature')['drop_feature'].agg(list).reset_index()
        return na_group

    def _permutate(self, feats):
        df = self.df_is_null_unique[feats]
        _ = df.apply(lambda x: ''.join(['Y' if i else 'N' for i in x]), 1)
        return np.column_stack((LabelEncoder().fit_transform(_), df.sum(1)))

    def permutate(self, n_jobs=16):
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            lst = pool.map(self._permutate, tqdm(self.na_group.drop_feature, 'Preprocessing ...'), chunksize=1)
        return np.column_stack(lst)
