<h1 align = "center">:rocket: iFeature :facepunch:</h1>

---
## `ifeature.featsub.FeatureSelector`
```python
fs = FeatureSelector(data, target)

# 单独使用
# 'all' = ['missing', 'single_unique', 'collinear', 'zero_importance', 'low_importance']
fs.remove(methods = 'missing')

# 一起使用
fs.identify_all( 
selection_params = {
    'missing_threshold': 0.8,
    'correlation_threshold': 0.98,
    'task': 'classification',
    'eval_metric': 'auc',
    'cumulative_importance': 0.99
    }
)
```

- `fs.identify_missing`
```python
fs.identify_missing(0.8)
fs.missing_stats
fs.record_missing
fs.plot_missing()
fs.ops['missing']
```

- `fs.identify_single_unique`
```python
fs.identify_single_unique()
fs.unique_stats
fs.record_single_unique
fs.plot_unique()
fs.ops['single_unique']
```

- `fs.identify_collinear`
```python
fs.identify_collinear(0.98)
fs.record_collinear
fs.plot_collinear()
fs.ops['collinear']
```

- `fs.identify_low_importance`
```python
"""
low_importance 方法借鉴了主成分分析(PCA)中的一种方法，其中仅保留维持一定方差比例(比如 95%)所需的主成分是很常见的做法。要纳入考虑的总重要度百分比基于同一思想。

只有当我们要用基于树的模型来做预测时，基于特征重要度的方法才真正有用。除了结果随机之外，基于重要度的方法还是一种黑箱方法，也就是说我们并不真正清楚模型认为某些特征无关的原因。如果使用这些方法，多次运行它们看到结果的改变情况，也许可以创建具有不同参数的多个数据集来进行测试!
"""
fs.identify_low_importance(cumulative_importance = 0.99)
fs.feature_importances
fs.record_low_importance
fs.plot_feature_importances(threshold = 0.99, plot_n = 12)
fs.ops['low_importance']

```

- `fs.identify_zero_importance`
```python
fs.identify_zero_importance(
    task = 'classification',  # 'regression'
    eval_metric = 'auc',
    n_iterations = 10,
    early_stopping = True
)
fs.record_zero_importance
fs.ops['zero_importance']
```

---

## **requires**
```python
python==3.6+
tqdm
lightgbm
seaborn
pandas
scikit-learn
```
