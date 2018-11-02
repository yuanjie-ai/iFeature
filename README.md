## 特征工程

### Install
```bash
pip install git+https://github.com/Jie-Yuan/iFeature.git
```

```
_clf = LGBMClassifier(n_estimators=1)
X = iris.data[:100, :]
y = iris.target[:100]
_clf.fit(X, y)
show_info = ['split_gain', 'internal_value', 'internal_count', 'leaf_count']
lgb.plot_tree(_clf.booster_, figsize=(60, 80), show_info=show_info)

model = _clf.booster_.dump_model()
tree_infos = model['tree_info'] # xgb_._Booster.get_dump()
```
