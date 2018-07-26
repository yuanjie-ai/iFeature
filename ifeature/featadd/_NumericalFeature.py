import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


class NumericalFeature(object):
    def __init__(self):
        pass

    @staticmethod
    def get_feats_poly(df, feats=None, degree=2, return_df=True):
        """PolynomialFeatures
        :param data: np.array or pd.DataFrame
        :param feats: columns names
        :param degree:
        :return: df
        """
        poly = PolynomialFeatures(degree, include_bias=False)
        df = poly.fit_transform(df[feats])

        if return_df:
            df = pd.DataFrame(df, columns=poly.get_feature_names(feats))
        return df

    @staticmethod
    def get_feats_agg_desc(df, group_col, agg_col):
        """data未聚合
        时间特征差分后当数值型特征
        """
        print("There are %s agg feats..." % len(agg_col))

        def q1(x):
            print('Compute %s ...' % x.name)
            return x.quantile(0.25)

        def q3(x): return x.quantile(0.75)

        def kurt(x): return x.kurt()

        def max_min(x): return x.max() - x.max()

        def q3_q1(x): return x.quantile(0.75) - x.quantile(0.25)

        def cv(x): return x.std() / (x.mean() + 10 ** -8)  # 变异系数

        def cv_reciprocal(x): return x.mean() / (x.std() + 10 ** -8)

        funcs = ['count', 'min', q1, 'mean', 'median', q3, 'max', 'sum', 'std', 'var', 'sem', 'skew', kurt, q3_q1,
                 max_min, cv, cv_reciprocal]

        df = df.groupby(group_col)[agg_col].agg(funcs)
        df.columns = ['_'.join(i) for i in df.columns]
        return df.reset_index()
