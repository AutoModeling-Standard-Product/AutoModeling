import pandas as pd
import numpy as np

import lightgbm as lgb
from bayes_opt import BayesianOptimization
from data_augmentation import weight
from gridsearch import *


# 沿用志奇gridsearch_params的前半部分，单独作为一个区分机构的方法
def split_by_org(data, label = 'new_target'):
    features = [f for f in data.columns if data[f].dtype != 'O' and f != label]
    tr_orgidx, val_orgidx, val_idx, tr_idx = {}, {}, [], []
    splitter = StratifiedShuffleSplit(n_splits=1, random_state=42, train_size=0.8)
    for org in data.new_org.unique():
        tmp_data = data[data.new_org==org].copy()
        for idx_tr, idx_val in splitter.split(tmp_data[feas], tmp_data['new_target']):
            tr_orgidx[org] = list(idx_tr)
            val_orgidx[org] = list(idx_val)
            val_idx += list(idx_val)
            tr_idx += list(idx_tr)
    data_tr, data_val = data.loc[tr_idx, ], data.loc[val_idx, ]
    X_tr, X_val, y_tr, y_val = data_tr[feas], data_val[feas], data_tr['new_target'], data_val['new_target']
    
    w_tr = pd.Series(np.ones(X_tr.shape[0]))

    return X_tr, X_val, y_tr, y_val, w_tr, tr_orgidx, val_orgidx


# 暂时不支持优化器并行，以训练器（此处为lgb）的并行来代替
def bayesian_search_params(
    data: pd.DataFrame, 
    fixed_params: dict,
    pbounds: dict,
    init_points: int, 
    max_iterations: int, 
    max_gap: float,
    min_ks: float,
    lgb_num_threads: int
):
    """
    :param data               输入的dataframe
    :param fixed_params       不参与优化的参数（如objective，metric等）  
    :param pbounds            待搜寻的参数项及对应的搜寻边界（tuple格式）
    :param init_points        随机初始评估点数量（建议为5）
    :param max_iterations     优化器最大迭代次数
    :param max_gap            业务约束：训练集（train）与验证集（val）的ks差值绝对值上限
    :param min_ks             业务约束：验证集（val）的ks下限
    :param lgb_num_threads    lgb训练并行线程数
    """
    X_tr, X_val, y_tr, y_val, w_tr, tr_orgidx, val_orgidx = split_by_org(data)
        
    def lgb_eval(**kwargs):
        train_params = {
            'objective': fixed_params.get('objective'),
            'metric': fixed_params.get('metric'),
            'learning_rate': kwargs.get('learning_rate'),
            'feature_fraction': np.clip(kwargs.get('feature_fraction'), 0, 1),
            'bagging_fraction': np.clip(kwargs.get('bagging_fraction'), 0, 1),
            'max_depth': int(kwargs.get('max_depth')),
            'lambda_l1': max(kwargs.get('lambda_l1'), 0),
            'lambda_l2': max(kwargs.get('lambda_l2'), 0),
            'min_split_gain': kwargs.get('min_split_gain'),
            'min_child_weight': kwargs.get('min_child_weight'),
            'verbose': fixed_params.get('verbose'),
            'seed': random_state,
            'boosting_type': fixed_params.get('boosting_type'),
            'num_threads': lgb_num_threads,
            'scale_pos_weight': weight
        }
        train_params = {k: v for k, v in train_params.items() if v}
        mean_tr_ks, mean_val_ks, mean_oos_ks = train_epoch(X_tr, X_val, y_tr, y_val, w_tr, tr_orgidx, val_orgidx, train_params, fobj=None)
        mean_ks_tr_val_diff = abs(mean_tr_ks - mean_val_ks) / mean_tr_ks
        if mean_ks_tr_val_diff <= max_gap and mean_val_ks > min_ks:
            return mean_val_ks
        else:
            return - sum(max(mean_ks_tr_val_diff, 0), max(min_ks - mean_val_ks, 0))

    optimizer = BayesianOptimization(
        f = lgb_eval,
        pbounds=pbounds,
        random_state=random_state
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=max_iterations
    )
    # 返回一个字典，包含两部分结果：‘target’（优化目标最优计算结果）和‘params（该计算结果对应的参数组合）’
    return optimizer.max
