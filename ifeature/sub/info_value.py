# coding=utf-8
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm


class InformationValue(object):
    def __init__(self, df: pd.DataFrame, label: str):
        """
        :param df:
        :param label: target name
        """
        assert label in df.columns

        self.label = label
        self.df = df.assign(_label=1 - df[label])
        self.cols = [col for col in df.columns if col != label]

        self.y1 = self.df[label].values.sum()
        self.y0 = self.df['_label'].values.sum()

    def iv(self, n_jobs=16):
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            ivs = pool.map(self._iv_func, tqdm(self.cols, 'Calculating ...'), chunksize=1)
        return pd.DataFrame(sorted(zip(ivs, self.cols), reverse=True), columns=['iv', 'feats'])

    def _iv_func(self, col):
        gr = self.df.groupby(col)
        gr1, gr0 = gr[self.label].sum().values + 1e-8, gr['_label'].sum().values + 1e-8
        good, bad = gr1 / self.y1, gr0 / self.y0
        woe = np.log(good / bad)
        iv = (good - bad) * woe
        return iv.sum()
