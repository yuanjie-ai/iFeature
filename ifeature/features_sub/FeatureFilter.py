# -*- coding: utf-8 -*-
"""
__title__ = 'fs'
__author__ = 'JieYuan'
__mtime__ = '2018/8/3'
"""

# from sklearn.feature_selection import VarianceThreshold

import numpy as np
import pandas as pd
from tqdm import tqdm

"""https://www.cnblogs.com/nolonely/p/6435083.html
1、为什么要做特征选择
在有限的样本数目下，用大量的特征来设计分类器计算开销太大而且分类性能差。
2、特征选择的确切含义
将高维空间的样本通过映射或者是变换的方式转换到低维空间，达到降维的目的，然后通过特征选取删选掉冗余和不相关的特征来进一步降维。
3、特征选取的原则
获取尽可能小的特征子集，不显著降低分类精度、不影响类分布以及特征子集应具有稳定适应性强等特点
"""


class FeatureFilter(object):
    """迭代
    高缺失率：
    低方差（高度重复值）：0.5%~99.5%分位数内方差为0的初筛
    高相关：特别高的初筛，根据重要性细筛
    低重要性：
    召回高IV：
    """

    def __init__(self, df, label=None):
        self.df = df
        self.label = label
        self.feats = df.columns.tolist()

        # 属性
        self.to_drop_missing = []
        self.to_drop_unique = []
        self.to_drop_variance = []
        self.to_drop_correlation = []
        self.corr_record = None
        self.to_drop_zero_importance = []
        self.to_drop_low_importance = []

    def filter_missing(self, feat_cols=None, threshold=0.95, as_na=None):
        """
        :param feat_cols:
        :param threshold:
        :param as_na: 比如把-99当成缺失值
        :return:
        """

        if feat_cols is None:
            feat_cols = self.feats

        # Calculate the fraction of missing in each column
        _df = self.df[feat_cols]

        # Sort with highest number of missing values on top
        # .sort_values('missing_fraction', ascending=False)
        s = (self.df == as_na).sum() if as_na else self.df.isnull().sum()
        missing_stats = (s / self.df.shape[0]) \
            .reset_index() \
            .rename(columns={'index': 'feature', 0: 'missing_fraction'})

        record_missing = missing_stats[lambda x: x.missing_fraction > threshold]

        to_drop = list(record_missing['feature'])

        # self.record_missing = record_missing

        self.to_drop_missing = to_drop

        print('%d features with greater than %0.2f missing values.\n' % (len(to_drop), threshold))

    def filter_unique(self, feat_cols=None):
        if feat_cols is None:
            feat_cols = self.feats

        """Finds features with only a single unique value. NaNs do not count as a unique value. """
        _df = self.df[feat_cols]

        # Calculate the unique counts in each column
        # unique_counts.sort_values('nunique', ascending=True)
        unique_counts = _df.nunique().reset_index().rename(columns={'index': 'feature', 0: 'nunique'})

        # Find the columns with only one unique count
        record_unique = unique_counts[lambda x: x['nunique'] == 1]

        to_drop = list(record_unique['feature'])

        # self.record_single_unique = record_unique

        self.to_drop_unique = to_drop

        print('%d features with a single unique value.\n' % len(to_drop))

    def filter_variance(self, feat_cols=None):
        if feat_cols is None:
            feat_cols = self.feats

        _df = self.df[feat_cols]

        to_drop = []
        for col in tqdm(feat_cols, 'Variance Filter'):
            if _df[col][lambda x: x.between(x.quantile(0.005), x.quantile(0.995))].var() == 0:
                to_drop.append(col)

        self.to_drop_variance = to_drop

        print('%d features with 0 variance in 0.5 ~ 99.5 quantile.\n' % len(to_drop))

    def filter_correlation(self, feat_cols=None, threshold=0.98):
        if feat_cols is None:
            feat_cols = self.feats

        print('Compute Corr Matrix ...')
        corr_matrix = self.df[feat_cols].corr().abs()

        # Extract the upper triangle of the correlation matrix
        upper = pd.DataFrame(np.triu(corr_matrix, 1), feat_cols, feat_cols)

        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in tqdm(upper.columns, 'Correlation Filter') if any(upper[column] > threshold)]

        self.to_drop_correlation = to_drop

        # Dataframe to hold correlated pairs
        # Iterate through the columns to drop to record pairs of correlated features
        corr_record = pd.DataFrame()
        for column in tqdm(to_drop, 'Correlation DataFrame'):
            cond = upper[column] > threshold
            corr_features = list(upper.index[cond])  # Find the correlated features
            corr_values = list(upper[column][cond])  # Find the correlated values
            drop_features = [column for _ in range(len(corr_features))]
            df_tmp = pd.DataFrame({'drop_feature': drop_features,
                                   'corr_feature': corr_features,
                                   'corr_value': corr_values})
            corr_record = corr_record.append(df_tmp, ignore_index=True, sort=False)

        self.corr_record = corr_record
        print('%d features with a  correlation coefficient greater than %0.2f.\n' % (len(to_drop), threshold))

    @property
    def to_drop_all(self):
        return self.to_drop_missing + self.to_drop_unique + self.to_drop_variance + self.to_drop_correlation + self.to_drop_zero_importance + self.to_drop_low_importance
