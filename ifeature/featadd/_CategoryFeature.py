# -*- coding: utf-8 -*-
"""
__title__ = '_get_feats'
__author__ = 'JieYuan'
__mtime__ = '2018/7/20'
"""
from sklearn.feature_extraction import text
from tqdm import tqdm


class CategoryFeature(object):
    @staticmethod
    def get_feats_vectors(X, vectorizer='TfidfVectorizer', tokenizer=None, ngram_range=(1, 1), max_features=None):
        """
        :param X: pd.Series
        :param vectorizer: 'TfidfVectorizer' or 'CountVectorizer'
        :param tokenizer: lambda x: x.split(',')
        :param ngram_range:
        :param max_features:
        :return:
        """
        vectorizer = text.__getattribute__(vectorizer)
        vectorizer = vectorizer(lowercase=False, tokenizer=tokenizer, ngram_range=ngram_range,
                                max_features=max_features)
        vectorizer.fit(X)
        return vectorizer

    @staticmethod
    def get_feats_desc(data, group='ID', feats=None):
        _gr = data.groupby(group)
        for col_name in tqdm(feats):
            funcs = ['count', 'nunique', 'max', 'min']
            _columns = {i: col_name + '_' + i for i in funcs}
            gr = _gr[col_name]

            def _func():
                df = gr.agg(funcs).reset_index()
                df[col_name + '_' + 'max_min'] = df['max'] - df['min']
                df[col_name + '_' + 'category_density'] = df['nunique'] / df['count']
                df[col_name + '_' + 'mode'] = gr.apply(lambda x: x.value_counts().index[0]).values
                return df.rename(columns=_columns)

            if col_name == feats[0]:
                df = _func()
            else:
                df = df.merge(_func(), 'left', group)
        return df.fillna(0)
