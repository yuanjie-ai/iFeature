import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures

class NumericalFeature(object):

    def __init__(self):
        pass

    @staticmethod
    def get_feats_poly(data, feats=None, degree=2, return_df=True):
        """PolynomialFeatures
        :param data: np.array or pd.DataFrame
        :param feats: columns names
        :param degree:
        :return: df
        """
        poly = PolynomialFeatures(degree, include_bias=False)
        data = poly.fit_transform(data[feats])

        if return_df:
            data = pd.DataFrame(data, columns=poly.get_feature_names(feats))
        return data

    @staticmethod
    def get_feats_desc(data, group='ID', feats=None):
        """data未聚合
        时间特征差分后当数值型特征
        """
        print("There are %s features..." % len(feats))

        for col_name in tqdm(feats, desc='get_feats_desc'):

            _columns = {i: col_name + '_' + i for i in ['count', 'mean', 'std', 'var', 'min', 'q1', 'median', 'q3', 'max']}
            gr = data.groupby(group)[col_name]

            def _func():
                # df = gr.describe().reset_index()
                df = gr.agg(['count', 'mean', 'std', 'var', 'min', 'median', 'max']).reset_index()
                df['q1'] = gr.apply(lambda x: x.quantile(0.25)).values
                df['q3'] = gr.apply(lambda x: x.quantile(0.75)).values
                df[col_name + '_' + 'max_min'] = df['max'] - df['min']
                df[col_name + '_' + 'q3_q1'] = df['q3'] - df['q1']
                df[col_name + '_' + 'kurt'] = gr.apply(pd.Series.kurt).values
                df[col_name + '_' + 'skew'] = gr.skew().values
                df[col_name + '_' + 'sem'] = gr.sem().values
                df[col_name + '_' + 'sum'] = gr.sum().values
                return df.rename(columns=_columns)

            if col_name == feats[0]:
                df = _func()
            else:
                df = df.merge(_func(), 'left', group).fillna(0)

        return df




