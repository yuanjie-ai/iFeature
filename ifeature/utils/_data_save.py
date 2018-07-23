# -*- coding: utf-8 -*-
"""
__title__ = '_data_save'
__author__ = 'JieYuan'
__mtime__ = '2018/7/22'
"""

import os
import pandas as pd

def data_save(df, hdf_path=None):
    assert hdf_path is not None
    print("save h5 ...")
    df.to_hdf(hdf_path, 'w', complib='blosc', complevel=6)

"""
if os.path.isfile(hdf_path):
    print("read h5 ...")
    pd.read_hdf(hdf_path)
"""
