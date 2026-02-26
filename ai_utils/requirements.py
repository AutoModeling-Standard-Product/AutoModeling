"""
依赖包导入模块
================

该模块统一导入项目所需的所有第三方库和内部模块。
采用集中导入方式，便于依赖管理和版本控制。

使用方式:
    from utils.requirements import *

依赖包说明:
    - pandas/numpy: 数据处理和数值计算
    - lightgbm: 梯度提升树模型
    - hyperopt: 贝叶斯优化调参
    - toad: 风控评分卡工具包
    - shap: 模型解释性分析
    - sklearn: 机器学习工具集
    - matplotlib/seaborn: 数据可视化

注意:
    - 当前依赖版本较老，建议后续升级
    - 部分依赖可能需要特定安装源
"""

##-- package dependences
## how to pip: !sudo /opt/conda3/bin/pip install -i http://192.168.101.40/pypi/simple --trusted-host 192.168.101.40 toad==0.0.64 imbalanced-learn==0.6.2 scikit-learn==0.24.2 shap==0.36.0 numba==0.48 llvmlite==0.31.0 mxnet==1.4.0 pathos

# ==================== 数据处理基础库 ====================
import pandas as pd
import numpy as np
import pickle

# ==================== 进度条显示 ====================
import tqdm

# ==================== 类型提示 ====================
from typing import Tuple

# ==================== 风控建模工具 ====================
import toad  # 风控评分卡工具包
import shap  # 模型解释性

# ==================== 模型持久化 ====================
import joblib
from joblib import load

# ==================== 可视化工具 ====================
import seaborn as sns
import matplotlib.pyplot as plt
from toad.plot import bin_plot, badrate_plot

# ==================== 时间处理 ====================
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ==================== 工具库 ====================
import itertools

# ==================== 机器学习库 ====================
import sklearn
from sklearn.model_selection import (
    train_test_split,      # 数据集切分
    GridSearchCV,          # 网格搜索
    StratifiedShuffleSplit,  # 分层抽样
    ParameterSampler       # 参数采样
)
from sklearn.metrics import (
    roc_auc_score,        # AUC指标
    roc_curve,            # ROC曲线
    auc,                  # AUC计算
    precision_recall_curve,  # PR曲线
    average_precision_score,  # 平均精度
    confusion_matrix,     # 混淆矩阵
    silhouette_score      # 轮廓系数
)
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.cluster import KMeans                # K-means聚类
from sklearn.decomposition import PCA             # 主成分分析
from sklearn.neighbors import *

# ==================== 梯度提升树模型 ====================
import lightgbm as lgb

# ==================== 超参数优化 ====================
from hyperopt import (
    fmin,        # 最小化函数
    tpe,         # TPE算法
    hp,          # 参数空间定义
    Trials,      # 试验记录
    space_eval,  # 参数空间评估
    STATUS_OK,   # 成功状态
    STATUS_FAIL  # 失败状态
)
from hyperopt.pyll import scope

# ==================== 统计分析 ====================
from scipy.stats import ks_2samp

# ==================== 时间和工具函数 ====================
import time
from functools import partial

# ==================== 多进程处理 ====================
import multiprocessing as multiprocessing
from multiprocessing import Pool, Manager, connection
# import utils.multiprocessing as multiprocessing  # Python 3.6兼容性补丁（备用）
# from utils.multiprocessing import Pool, Manager, connection

# ==================== 其他工具 ====================
import ast  # 抽象语法树，用于解析字符串形式的列表/字典
import os   # 操作系统接口
from pathlib import Path  # 路径处理
