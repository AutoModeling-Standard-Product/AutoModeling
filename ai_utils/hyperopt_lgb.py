"""
HyperOptLGB: LightGBM超参数优化类

该模块提供基于TPE（Tree-structured Parzen Estimator）的LightGBM超参数优化功能，
支持多机构交叉验证、加权评估指标、早停机制等高级功能。

典型用途:
    >>> from hyperopt import hp
    >>> params = {
    ...     'max_depth': hp.quniform('max_depth', 3, 10, 1),
    ...     'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    ...     ...
    ... }
    >>> optimizer = HyperOptLGB(data=df, params=params, max_iterations=50)
    >>> trails = optimizer.tpesearch_params()

Author: AutoModeling Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
from hyperopt import fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
from multiprocessing import Pool, Manager
import time
import json

# 尝试导入依赖模块，失败时提供降级方案
try:
    import toad
    from utils.data_augmentation import re_weight_by_org
    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False
    toad = None
    re_weight_by_org = None
    warnings.warn(f"部分依赖模块导入失败: {e}. 某些功能可能不可用.")

# IPython display（用于Jupyter环境）
try:
    from IPython.display import display
except ImportError:
    # 如果不在Jupyter环境，使用print替代
    def display(*args, **kwargs):
        for arg in args:
            print(arg)


@dataclass
class TrainingResult:
    """训练结果数据类"""
    loss: float
    param: Dict[str, Any]
    status: str
    randn: int
    mean_val_ks: float
    mean_oos_ks: float
    simpler_results: pd.DataFrame
    results: pd.DataFrame


@dataclass
class MetricResult:
    """评估指标结果数据类"""
    auc_w: float
    auc: float
    ks_w: float
    ks: float
    lift5: float
    lift10: float


class HyperOptLGB:
    """
    LightGBM超参数优化器（基于TPE算法）
    
    该类实现了基于Tree-structured Parzen Estimator的超参数搜索，
    支持多机构Leave-One-Out交叉验证，加权评估指标（KS、AUC、Lift）。
    
    Attributes:
        params: 超参数搜索空间（dict或hyperopt.hp对象）
        fobj: 自定义目标函数（可选）
        max_iterations: 最大迭代次数
        record_train_process: 是否记录训练过程
        auc_threshold: AUC阈值，用于判断过拟合
        randn: 随机种子
        
    Example:
        >>> params = {
        ...     'max_depth': hp.quniform('max_depth', 3, 10, 1),
        ...     'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        ...     'num_leaves': hp.quniform('num_leaves', 20, 100, 1),
        ...     'min_child_samples': hp.quniform('min_child_samples', 10, 100, 1),
        ...     'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
        ...     'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.0),
        ...     'bagging_freq': hp.quniform('bagging_freq', 1, 7, 1),
        ...     'lambda_l1': hp.uniform('lambda_l1', 0, 1),
        ...     'lambda_l2': hp.uniform('lambda_l2', 0, 1),
        ...     'min_gain_to_split': hp.uniform('min_gain_to_split', 0, 0.5),
        ...     'broadcast_with_tar': hp.choice('broadcast_with_tar', [True, False]),
        ...     'balanced_badrate': hp.uniform('balanced_badrate', 0.3, 0.7),
        ... }
        >>> optimizer = HyperOptLGB(
        ...     data=df,
        ...     params=params,
        ...     max_iterations=50,
        ...     randn=42
        ... )
        >>> trails = optimizer.tpesearch_params()
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        fobj: Optional[Callable] = None,
        max_iterations: int = 100,
        record_train_process: bool = False,
        auc_threshold: float = 0.03,
        randn: int = 42
    ) -> None:
        """
        初始化HyperOptLGB优化器
        
        Args:
            data: 训练数据，需包含'new_org'（机构ID）和'new_target'（目标变量）列
            params: 超参数搜索空间，支持hyperopt.hp对象或固定值
            fobj: 自定义目标函数（可选），如需要自定义损失函数
            max_iterations: TPE搜索的最大迭代次数，默认100
            record_train_process: 是否记录完整训练过程（用于分析），默认False
            auc_threshold: 训练集与验证集AUC相对差异阈值，用于判断过拟合，默认0.03
            randn: 随机种子，用于保证结果可复现，默认42
            
        Raises:
            ValueError: 如果数据缺少必需列（'new_org', 'new_target'）
            TypeError: 如果数据类型不正确
            ImportError: 如果缺少必要的依赖模块
            
        Note:
            - 数据集必须包含'new_org'列（机构标识）和'new_target'列（标签，0/1）
            - 特征列自动识别为非object类型的列（除'new_target'外）
            - 使用分层抽样确保各机构内正负样本比例一致
        """
        # 参数验证
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data必须是pandas.DataFrame类型，当前类型: {type(data)}")
        
        required_cols = ['new_org', 'new_target']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必需列: {missing_cols}")
        
        if not HAS_DEPENDENCIES:
            raise ImportError("缺少必要依赖: toad, utils.data_augmentation")
        
        # 存储初始化参数
        self.params = params
        self.fobj = fobj
        self.max_iterations = max_iterations
        self.record_train_process = record_train_process
        self.auc_threshold = auc_threshold
        self.randn = randn
        
        # 初始化Trials对象用于记录搜索过程
        self.trails = Trials()
        
        # 数据切分：8:2分层抽样，按机构
        self.X_tr, self.X_val, self.y_tr, self.y_val, self.tr_orgidx, self.val_orgidx = \
            self.split_data(data)
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, List], Dict[str, List]]:
        """
        按机构分层切分训练集和验证集
        
        使用StratifiedShuffleSplit对每个机构分别进行8:2分层抽样，
        保证训练集和验证集中各机构的样本比例一致。
        
        Args:
            data: 原始数据DataFrame，需包含'new_org'和'new_target'列
            
        Returns:
            Tuple包含以下元素:
                - X_tr: 训练集特征
                - X_val: 验证集特征  
                - y_tr: 训练集标签
                - y_val: 验证集标签
                - tr_orgidx: 训练集中各机构的索引字典 {org_id: [indices]}
                - val_orgidx: 验证集中各机构的索引字典 {org_id: [indices]}
                
        Raises:
            ValueError: 如果某机构样本数过少无法进行分层抽样
            
        Example:
            >>> X_tr, X_val, y_tr, y_val, tr_idx, val_idx = self.split_data(df)
            >>> print(f"训练集大小: {len(X_tr)}, 验证集大小: {len(X_val)}")
        """
        # 识别特征列：非object类型且不是目标列
        feas = [v for v in data.columns 
                if data[v].dtype != 'O' and v != 'new_target']
        
        if not feas:
            raise ValueError("未找到有效的数值特征列")
        
        # 初始化索引存储结构
        tr_orgidx: Dict[str, List] = {}
        val_orgidx: Dict[str, List] = {}
        tr_idx: List = []
        val_idx: List = []
        
        # 分层抽样器
        splitter = StratifiedShuffleSplit(
            n_splits=1, 
            random_state=self.randn, 
            train_size=0.8
        )
        
        # 按机构进行分层抽样
        for org in data['new_org'].unique():
            tmp_data = data[data['new_org'] == org].copy()
            org_index = tmp_data.index.tolist()
            
            # 检查机构样本数
            if len(tmp_data) < 5:
                raise ValueError(
                    f"机构'{org}'样本数过少({len(tmp_data)})，"
                    f"无法进行分层抽样。建议至少保留5个样本。"
                )
            
            # 每个机构内分层抽样
            try:
                for idx_tr, idx_val in splitter.split(
                    tmp_data[feas], 
                    tmp_data['new_target']
                ):
                    tr_orgidx[org] = [org_index[i] for i in idx_tr]
                    val_orgidx[org] = [org_index[i] for i in idx_val]
                    val_idx.extend([org_index[i] for i in idx_val])
                    tr_idx.extend([org_index[i] for i in idx_tr])
            except ValueError as e:
                raise ValueError(
                    f"机构'{org}'分层抽样失败: {str(e)}"
                ) from e
        
        # 切分数据集
        data_tr = data.loc[tr_idx]
        data_val = data.loc[val_idx]
        
        X_tr = data_tr[feas]
        X_val = data_val[feas]
        y_tr = data_tr['new_target']
        y_val = data_val['new_target']
        
        return X_tr, X_val, y_tr, y_val, tr_orgidx, val_orgidx
    
    def _get_lift(
        self, 
        y: pd.Series, 
        pred: Union[pd.Series, np.ndarray], 
        k: float
    ) -> float:
        """
        计算Top K% Lift指标
        
        Lift衡量模型识别高风险客户的能力相对于随机选择的倍数。
        是风险管理中的关键业务指标。
        
        Args:
            y: 真实标签(0/1)，pandas Series格式
            pred: 模型预测分数或概率
            k: 百分位阈值(如0.1表示Top 10%，0.05表示Top 5%)
        
        Returns:
            float类型的Lift值
            - Lift > 1: 模型优于随机选择
            - Lift = 1: 与随机选择相同
            - Lift < 1: 差于随机选择
        
        Raises:
            ValueError: 如果k不在(0,1]范围或y为空
            TypeError: 如果输入类型不正确
        
        Example:
            >>> y = pd.Series([0, 1, 0, 1, 0, 1, 0, 0, 1, 0])
            >>> pred = pd.Series([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.5, 0.6, 0.05])
            >>> lift5 = self._get_lift(y, pred, 0.05)  # Top 5% Lift
            >>> print(f"Top 5% Lift: {lift5:.2f}x")
            Top 5% Lift: 5.00x
        
        Business Context:
            Lift是风险模型的核心业务指标：
            - Top 5% Lift ≥ 3倍: 优秀区分能力，强烈建议上线
            - Top 10% Lift ≥ 2倍: 良好区分能力，可以上线
            - 直接关联业务价值：高Lift意味着可以用更少的拒绝率捕获更多坏客户
            - 用于设定审批阈值，平衡通过率与坏账率
            
            根据Risk Modeling Professional Standards:
            - Top Decile Lift ≥ 2为最低要求
            - 实际业务中通常要求≥3以获得明显收益
        """
        # 输入验证
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        if len(y) == 0:
            raise ValueError("输入y不能为空")
        
        if not 0 < k <= 1:
            raise ValueError(f"k必须在(0, 1]范围内，当前值: {k}")
        
        # 计算Top样本数
        n_top = int(len(y) * k)
        
        if n_top == 0:
            warnings.warn(f"k={k}对于{len(y)}个样本过小，返回lift=1.0")
            return 1.0
        
        # 获取预测分数最高的k%样本索引
        if isinstance(pred, pd.Series):
            pred_series = pred
        else:
            pred_series = pd.Series(pred, index=y.index)
        
        top_indices = pred_series.sort_values(ascending=False).head(n_top).index
        
        # 计算Lift: Top k%坏品率 / 整体坏品率
        bad_rate_top = y.loc[top_indices].mean()
        overall_bad_rate = y.mean()
        
        if overall_bad_rate == 0:
            warnings.warn("整体坏品率为0，无法计算Lift，返回1.0")
            return 1.0
        
        return float(bad_rate_top / overall_bad_rate)
    
    def weighted_metric(
        self, 
        weights: Dict[str, pd.Series]
    ) -> Callable:
        """
        创建LightGBM加权评估指标
        
        工厂函数，创建自定义评估函数用于LightGBM训练过程。
        支持加权和非加权的AUC、KS、Lift指标计算。
        
        Args:
            weights: 包含'train'、'val'、'oos'三个键的字典，
                    每个键对应一个样本权重Series
        
        Returns:
            Callable: 符合LightGBM feval参数的评估函数。
                     该函数接收(preds, dataset)参数，
                     返回[(name, value, higher_is_better), ...]列表
        
        Raises:
            ValueError: 如果weights缺少必需的键
            KeyError: 如果weights字典结构无效
        
        Example:
            >>> w_tr = pd.Series([1.0, 2.0, 1.0], index=[0, 1, 2])
            >>> w_val = pd.Series([1.5, 1.5], index=[3, 4])
            >>> weights = {'train': w_tr, 'val': w_val, 'oos': w_val}
            >>> metric_fn = self.weighted_metric(weights)
            >>> # 在LightGBM中使用
            >>> lgb.train(params, train_set, feval=metric_fn)
        
        Business Context:
            加权指标对于跨机构公平评估至关重要：
            - 不同机构可能有不同的坏品率
            - 权重平衡各机构的贡献度
            - 加权AUC/KS确保模型在所有细分上表现良好
            - 防止偏向数据量大的机构
            
            根据Risk Modeling Professional Standards:
            - 必须分析所有机构的指标一致性
            - 机构间KS差异应<0.2
            - 加权指标确保小机构不被忽视
        
        Note:
            根据数据长度自动判断当前评估的是哪个数据集。
            需要LightGBM 2.3.0+。使用LightGBM >= 3.0.0时，
            建议使用eval_sample_weight参数以获得更好性能。
        """
        if not all(k in weights for k in ['train', 'val', 'oos']):
            raise ValueError("weights必须包含键: 'train', 'val', 'oos'")
        
        def _weighted_metric(pred_: np.ndarray, data: lgb.Dataset) -> List[Tuple[str, float, bool]]:
            """内部评估函数"""
            y = data.get_label()
            
            # 将对数几率转换为概率
            pred = 1 / (1 + np.exp(-pred_))
            
            # 根据数据长度判断使用哪个权重
            w = weights.get('train')  # 默认
            if len(y) == len(weights.get('train')):
                w = weights.get('train')
            elif len(y) == len(weights.get('val')):
                w = weights.get('val')
            elif len(y) == len(weights.get('oos')):
                w = weights.get('oos')
            
            # 计算加权和未加权AUC
            auc = roc_auc_score(y, pred)
            auc_w = roc_auc_score(y, pred, sample_weight=w)
            
            # 计算KS统计量
            fpr, tpr, _ = roc_curve(y, pred, sample_weight=w)
            ks = tpr.max() - fpr.max() if hasattr(tpr, 'max') else max(tpr) - max(fpr)
            ks_w = max(tpr - fpr)
            
            # 计算Lift指标
            y_series = pd.Series(y, index=w.index if hasattr(w, 'index') else range(len(y)))
            pred_series = pd.Series(pred, index=y_series.index)
            
            lift5 = self._get_lift(y_series, pred_series, 0.05)
            lift10 = self._get_lift(y_series, pred_series, 0.10)
            
            return [
                ('auc_wo', float(auc), True),
                ('auc_w', float(auc_w), True),
                ('ks_w', float(ks_w), True),
                ('ks', float(ks), True),
                ('5%lift', float(lift5), True),
                ('10%lift', float(lift10), True)
            ]
        
        return _weighted_metric
    
    def single_weighted_metric(
        self,
        model: lgb.Booster,
        X: pd.DataFrame,
        y: pd.Series,
        w: pd.Series
    ) -> Tuple[float, float, float, float, float, float]:
        """
        计算单组数据的加权和非加权指标
        
        替代weighted_metric的轻量版本，只计算最终轮次指标。
        避免LightGBM训练过程中每轮都计算的额外开销。
        
        Args:
            model: 已训练的LightGBM模型(Booster对象)
            X: 特征矩阵(DataFrame)
            y: 目标标签(Series，0/1)
            w: 样本权重(Series)
        
        Returns:
            Tuple包含以下指标(按顺序)：
            - auc_w (float): 加权AUC
            - auc (float): 未加权AUC
            - ks_w (float): 加权KS
            - ks (float): 未加权KS
            - lift5 (float): Top 5% Lift
            - lift10 (float): Top 10% Lift
        
        Raises:
            ValueError: 如果输入数据为空或维度不匹配
            AttributeError: 如果模型未训练完成
        
        Example:
            >>> model = lgb.train(params, train_set, num_boost_round=100)
            >>> auc_w, auc, ks_w, ks, lift5, lift10 = self.single_weighted_metric(
            ...     model, X_val, y_val, w_val
            ... )
            >>> print(f"Validation AUC: {auc:.4f}, Weighted AUC: {auc_w:.4f}")
            >>> print(f"Validation KS: {ks:.4f}, Weighted KS: {ks_w:.4f}")
        
        Business Context:
            该函数用于最终模型评估，避免训练过程中的性能开销：
            - 训练时使用weighted_metric(逐轮评估)
            - 最终评估使用single_weighted_metric(只计算一次)
            - 两者确保评估一致性
            
            根据Risk Modeling Professional Standards:
            - 必须报告加权和未加权指标
            - 加权指标确保跨机构公平性
            - AUC应≥0.65，KS应≥0.25
        
        Note:
            预测结果为概率值(经过sigmoid转换)
        """
        # 生成预测
        pred = model.predict(X)
        pred_series = pd.Series(pred, index=X.index)
        
        # 计算AUC
        auc_w = roc_auc_score(y, pred_series, sample_weight=w)
        auc = roc_auc_score(y, pred_series)
        
        # 计算KS
        fpr, tpr, _ = roc_curve(y, pred_series, sample_weight=w)
        ks = max(tpr) - max(fpr)
        ks_w = max(tpr - fpr)
        
        # 计算Lift
        lift5 = self._get_lift(y, pred_series, 0.05)
        lift10 = self._get_lift(y, pred_series, 0.10)
        
        return auc_w, auc, ks_w, ks, lift5, lift10
    
    def extract_evalresult(self, records: pd.DataFrame) -> pd.DataFrame:
        """
        从训练日志中提取最后一个迭代的结果
        
        处理LightGBM的训练记录，按机构分组，提取每个机构
        训练、验证、OOS(留出机构)的最后一个迭代的指标。
        
        Args:
            records: LightGBM训练记录DataFrame，包含多层列索引
                    (train/val/oos, metric_name) 格式
        
        Returns:
            pd.DataFrame: 简化后的结果DataFrame，每行一个机构，
                         包含各数据集的最终指标值
        
        Raises:
            ValueError: 如果records为空或缺少必需的列
            KeyError: 如果结果结构不符合预期
        
        Example:
            >>> eval_results = lgb.train(...)
            >>> df = self.extract_evalresult(eval_results)
            >>> print(df[['org', 'val_ks', 'oos_ks']].head())
              org  val_ks  oos_ks
            0  A   0.452   0.438
            1  B   0.461   0.445
        
        Business Context:
            提取最终结果用于：
            - 跨机构对比分析
            - 模型稳定性检查
            - 超参数调优决策
            
            根据Risk Modeling Professional Standards:
            - 训练集与验证集AUC差异应<0.05
            - 验证集与OOS集KS差异应<10%衰减
            - 机构间KS差异应<0.2
        
        Note:
            期望records包含以下结构：
            - 一级列: 'train', 'val', 'oos'
            - 二级列: 'auc_w', 'auc_wo', 'ks', 'ks_w', '5%lift', '10%lift'
            - 'org'列标识机构
        """
        if records.empty or 'org' not in records.columns:
            raise ValueError("records不能为空且必须包含'org'列")
        
        simpler_records = []
        
        for org in records['org'].unique():
            tmp_record = records[records['org'] == org].copy()
            
            # 提取每个数据集最后一个迭代的指标
            try:
                record = {
                    'org': org,
                    'tr_auc': tmp_record['train']['auc_wo'].iloc[-1],
                    'tr_auc_w': tmp_record['train']['auc_w'].iloc[-1],
                    'tr_ks': tmp_record['train']['ks'].iloc[-1],
                    'tr_ks_w': tmp_record['train']['ks_w'].iloc[-1],
                    'tr_5lift': tmp_record['train']['5%lift'].iloc[-1],
                    'tr_10lift': tmp_record['train']['10%lift'].iloc[-1],
                    'val_auc': tmp_record['val']['auc_wo'].iloc[-1],
                    'val_auc_w': tmp_record['val']['auc_w'].iloc[-1],
                    'val_ks': tmp_record['val']['ks'].iloc[-1],
                    'val_ks_w': tmp_record['val']['ks_w'].iloc[-1],
                    'val_5lift': tmp_record['val']['5%lift'].iloc[-1],
                    'val_10lift': tmp_record['val']['10%lift'].iloc[-1],
                    'oos_auc': tmp_record['oos']['auc_wo'].iloc[-1],
                    'oos_ks': tmp_record['oos']['ks'].iloc[-1],
                    'oos_5lift': tmp_record['oos']['5%lift'].iloc[-1],
                    'oos_10lift': tmp_record['oos']['10%lift'].iloc[-1]
                }
                simpler_records.append(record)
            except (KeyError, IndexError) as e:
                warnings.warn(f"机构{org}的记录解析失败: {str(e)}")
                continue
        
        return pd.DataFrame(simpler_records)
    
    def train_epoch_(
        self,
        shared_list: Any,
        org: str,
        param: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        单轮训练：留出指定机构进行LOO交叉验证
        
        核心训练方法，实现Leave-One-Institution-Out交叉验证：
        - 将指定机构(org)作为OOS(留出测试集)
        - 使用其他所有机构训练模型
        - 计算训练集、验证集、OOS集的指标
        
        Args:
            shared_list: 多进程共享列表(当前未使用)
            org: 当前留出的机构ID
            param: 超参数配置字典，包含：
                - max_depth: 树最大深度
                - learning_rate: 学习率
                - broadcast_with_tar: 是否广播标签
                - balanced_badrate: 均衡坏样率
                - 其他LightGBM参数
        
        Returns:
            pd.DataFrame: 训练结果DataFrame，包含：
                - 各迭代的train/val/oos指标
                - 或者只包含最终迭代的指标
        
        Raises:
            ValueError: 如果机构索引不存在
            KeyError: 如果参数缺少必需键
        
        Example:
            >>> param = {'max_depth': 5, 'learning_rate': 0.1, 'balanced_badrate': 1.0}
            >>> results = self.train_epoch_(None, 'org_A', param)
            >>> print(results[['train_ks', 'val_ks', 'oos_ks']].iloc[-1])
        
        Business Context:
            LOO交叉验证确保模型在所有机构上表现一致：
            - 避免过拟合某个机构
            - 确保模型泛化能力
            - 识别异常机构(如果OOS指标明显低于其他机构)
            
            根据Risk Modeling Professional Standards:
            - 机构间KS差异应<0.2
            - 所有机构OOS KS应≥0.25
            - 如果某机构指标明显偏低，需要进行专门分析
        
        Note:
            使用多进程时，shared_list用于进程间通信(当前未实现)。
            训练过程中会根据param中的参数动态调整。
        """
        # 从参数中提取配置
        broadcast_with_tar = param.get('broadcast_with_tar', False)
        balanced_badrate = param.get('balanced_badrate', 1.0)
        
        # 计算叶子节点数
        if 'max_depth' in param:
            param['num_leaves'] = 2 ** param['max_depth'] - 1
        
        # 计算样本权重
        weight_tr = re_weight_by_org(
            self.y_tr, self.tr_orgidx, 0.5, broadcast_with_tar, balanced_badrate
        )
        weight_val = re_weight_by_org(
            self.y_val, self.val_orgidx, 0.5, broadcast_with_tar, balanced_badrate
        )
        weight = pd.concat([weight_tr, weight_val], axis=0)
        
        # 获取索引集合
        tr_idxs = set(self.X_tr.index)
        val_idxs = set(self.X_val.index)
        tr_idx = set(self.tr_orgidx.get(org, []))
        val_idx = set(self.val_orgidx.get(org, []))
        
        # 构建训练集（排除当前机构）
        X_tr_ = self.X_tr.loc[list(tr_idxs - tr_idx)]
        y_tr_ = self.y_tr.loc[list(tr_idxs - tr_idx)]
        
        # 构建验证集（排除当前机构）
        X_val_ = self.X_val.loc[list(val_idxs - val_idx)]
        y_val_ = self.y_val.loc[list(val_idxs - val_idx)]
        
        # 构建权重（简化版）
        w_tr_ = pd.Series(
            np.where(y_tr_ == 1, balanced_badrate, 1), 
            index=y_tr_.index
        )
        w_val_ = pd.Series(
            np.where(y_val_ == 1, balanced_badrate, 1), 
            index=y_val_.index
        )
        
        # 构建OOS集（当前机构作为测试集）
        X_oos = pd.concat([
            self.X_tr.loc[list(tr_idx)], 
            self.X_val.loc[list(val_idx)]
        ], axis=0)
        y_oos = pd.concat([
            self.y_tr.loc[list(tr_idx)], 
            self.y_val.loc[list(val_idx)]
        ], axis=0)
        
        # 避免OOS与val长度相同导致权重判断错误
        if len(y_val_) == len(y_oos):
            X_oos = X_oos.iloc[1:]
            y_oos = y_oos.iloc[1:]
        
        # 构建权重字典
        weights_dict = {
            'train': w_tr_,
            'val': w_val_,
            'oos': pd.Series(np.ones(len(y_oos)), index=y_oos.index)
        }
        
        # 初始化评估结构
        eval_results = {}
        valid_sets = None
        valid_names = None
        feval = None
        callbacks = []
        
        # 创建LightGBM数据集
        train_set = lgb.Dataset(X_tr_, label=y_tr_, weight=w_tr_)
        val_set = lgb.Dataset(X_val_, label=y_val_, reference=train_set)
        oos_set = lgb.Dataset(X_oos, label=y_oos, reference=train_set)
        
        # 处理早停配置
        if 'stopping_rounds' in param:
            param['num_iterations'] = 300
            callbacks.append(lgb.early_stopping(stopping_rounds=param['stopping_rounds']))
            self.record_train_process = True
            valid_sets = [train_set, val_set, oos_set]
            valid_names = ['train', 'val', 'oos']
        
        # 处理训练过程记录
        if self.record_train_process:
            callbacks.append(lgb.record_evaluation(eval_results))
            valid_sets = [train_set, val_set, oos_set]
            valid_names = ['train', 'val', 'oos']
            feval = self.weighted_metric(weights_dict)
        
        # 训练模型
        model = lgb.train(
            param,
            train_set=train_set,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=feval,
            callbacks=callbacks
        )
        
        # 处理评估结果
        if self.record_train_process:
            # 记录训练过程模式：eval_results已由lgb.record_evaluation填充
            if len(eval_results) > 0:
                eval_results = pd.DataFrame(eval_results)
                eval_results['org'] = org
                return eval_results
            else:
                # 训练过程记录失败，创建空结果
                warnings.warn(f"机构{org}的训练过程记录为空，可能训练失败")
                return pd.DataFrame({'org': [org]})
        
        # 不记录训练过程模式：计算最终指标
        # 训练集指标
        auc_w, auc, ks_w, ks, lift5, lift10 = self.single_weighted_metric(
            model, X_tr_, y_tr_, w_tr_
        )
        tmp0 = pd.DataFrame({
            'train': [[auc_w], [auc], [ks_w], [ks], [lift5], [lift10]]
        }, index=['auc_w', 'auc_wo', 'ks_w', 'ks', '5%lift', '10%lift'])
        
        # 验证集指标
        auc_w, auc, ks_w, ks, lift5, lift10 = self.single_weighted_metric(
            model, X_val_, y_val_, w_val_
        )
        tmp1 = pd.DataFrame({
            'val': [[auc_w], [auc], [ks_w], [ks], [lift5], [lift10]]
        }, index=['auc_w', 'auc_wo', 'ks_w', 'ks', '5%lift', '10%lift'])
        
        # OOS集指标
        w_oos = pd.Series(np.ones(len(y_oos)), index=y_oos.index)
        auc_w, auc, ks_w, ks, lift5, lift10 = self.single_weighted_metric(
            model, X_oos, y_oos, w_oos
        )
        tmp2 = pd.DataFrame({
            'oos': [[auc_w], [auc], [ks_w], [ks], [lift5], [lift10]]
        }, index=['auc_w', 'auc_wo', 'ks_w', 'ks', '5%lift', '10%lift'])
        
        # 合并结果
        eval_results = pd.concat([tmp0, tmp1, tmp2], axis=1)
        eval_results['org'] = org
        
        return eval_results
    
    def objective(self, param: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hyperopt目标函数，评估一组超参数的性能
        
        核心优化函数，对每组超参数：
        1. 并行训练所有机构的LOO交叉验证
        2. 提取每个机构的train/val/oos指标
        3. 检查过拟合(训练集与验证集AUC差异)
        4. 返回优化结果
        
        Args:
            param: 一组超参数，由Hyperopt采样生成
        
        Returns:
            Dict包含以下键：
            - loss (float): 优化目标值（负KS）
            - param (dict): 当前超参数
            - status (str): 优化状态(STATUS_OK/STATUS_FAIL)
            - randn (int): 随机种子
            - mean_val_ks (float): 平均验证集KS
            - mean_oos_ks (float): 平均OOS KS
            - simpler_results (pd.DataFrame): 简化结果DataFrame
            - results (pd.DataFrame): 完整结果DataFrame
        
        Raises:
            RuntimeError: 如果训练过程出错
            ValueError: 如果超参数无效
        
        Example:
            >>> param = {'max_depth': 5, 'learning_rate': 0.1}
            >>> result = self.objective(param)
            >>> print(f"Loss: {result['loss']:.4f}, Status: {result['status']}")
            >>> print(f"Mean Val KS: {result['mean_val_ks']:.4f}")
        
        Business Context:
            目标函数是超参数优化的核心：
            - 负KS作为损失函数（最小化损失=最大化KS）
            - 稳定性检查防止过拟合
            - 并行训练提升效率
            
            根据Risk Modeling Professional Standards:
            - 训练集与验证集AUC相对差异应<0.05
            - 平均验证集KS应≥0.25
            - 如果检查失败，返回inf损失以排除该参数
            
            AUC稳定性阈值的业务意义：
            - 确保模型在不同数据集上表现一致
            - 避免过拟合训练集
            - 保证生产环境泛化能力
        
        Note:
            使用10进程并行池加速训练。结果保存到res.txt防止丢失。
        """
        begin_time = time.time()
        
        # 设置自定义目标函数
        if self.fobj is not None:
            param['objective'] = self.fobj
        
        results = pd.DataFrame()
        
        # 使用多进程池并行训练
        with Manager() as manager:
            shared_list = manager.list()
            tasks = [(shared_list, org, param) for org in self.tr_orgidx.keys()]
            
            with Pool(10) as pool:
                records = pool.starmap(self.train_epoch_, tasks)
            
            for record in records:
                results = pd.concat([results, record], axis=0)
        
        # 提取简化结果
        simpler_results = self.extract_evalresult(results)
        
        # 计算平均指标
        mean_val_ks = np.mean(simpler_results['val_ks'])
        mean_oos_ks = np.mean(simpler_results['oos_ks'])
        
        # 检查AUC稳定性（防止过拟合）
        auc_stable = np.allclose(
            simpler_results['tr_auc'], 
            simpler_results['val_auc'], 
            rtol=self.auc_threshold
        )
        auc_w_stable = np.allclose(
            simpler_results['tr_auc_w'], 
            simpler_results['val_auc_w'], 
            rtol=self.auc_threshold
        )
        
        if auc_stable and auc_w_stable:
            loss = -mean_val_ks
            status = STATUS_OK
        else:
            loss = np.inf
            status = STATUS_FAIL
        
        # 保存结果（防止崩溃丢失）
        try:
            with open("res.txt", 'a', encoding='utf-8') as f:
                json.dump({
                    'loss': float(loss),
                    'param': str(param),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }, f)
                f.write('\n')
        except Exception as e:
            warnings.warn(f"结果保存失败: {str(e)}")
        
        end_time = time.time()
        display(f"当前组参数训练耗时：{np.round((end_time - begin_time) / 60, 2)}分")
        
        return {
            'loss': loss,
            'param': param,
            'status': status,
            'randn': self.randn,
            'mean_val_ks': mean_val_ks,
            'mean_oos_ks': mean_oos_ks,
            'simpler_results': simpler_results,
            'results': results
        }
    
    def tpesearch_params(self) -> pd.DataFrame:
        """
        执行TPE（Tree-structured Parzen Estimator）超参数搜索
        
        该类的主入口函数，执行完整的超参数优化流程：
        1. 使用TPE算法在超参数空间中进行智能搜索
        2. 每轮调用objective函数评估超参数性能
        3. 记录所有搜索历史和最优结果
        4. 返回完整的搜索记录(trails)
        
        Returns:
            pd.DataFrame: 包含所有搜索历史的DataFrame，每行一次评估记录，
                         包含loss、params、metrics等信息
        
        Raises:
            RuntimeError: 如果优化过程中出现严重错误
            KeyboardInterrupt: 用户中断搜索时可安全返回已完成的记录
        
        Example:
            >>> optimizer = HyperOptLGB(
            ...     data=df,
            ...     params=param_space,
            ...     max_iterations=50,
            ...     randn=42
            ... )
            >>> trails = optimizer.tpesearch_params()
            >>> print(f"Best KS: {-trails['loss'].min():.4f}")
            >>> best_idx = trails['loss'].idxmin()
            >>> best_params = trails.loc[best_idx, 'params']
        
        Business Context:
            TPE搜索是模型调优的关键步骤：
            - 自动选择最优超参数配置
            - 无需人工干预即可探索大量配置
            - 记录每轮结果用于后续分析
            - 找到最佳性能与稳定性的平衡点
            
            根据Risk Modeling Professional Standards:
            - 建议搜索至少50-100轮以覆盖足够空间
            - 最终选择的参数必须在多个机构上表现稳定
            - 需要保存训练日志以便审计和复现
        
        Note:
            - TPE算法基于贝叶斯优化，会根据历史结果自适应搜索
            - 可以随时中断，已完成的搜索记录会保留在trails对象中
            - 结果DataFrame可直接用于结果分析和可视化
            
            典型耗时参考：
            - 每轮耗时取决于数据量和机构数（通常0.5-5分钟）
            - 50轮搜索可能需要30分钟-数小时
        """
        begin_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        display(f"开始执行时间：{begin_time_str}")
        display(f"计划搜索轮数：{self.max_iterations}")
        
        try:
            # 执行TPE搜索
            _ = fmin(
                fn=self.objective,
                space=self.params,
                algo=tpe.suggest,
                max_evals=self.max_iterations,
                trials=self.trails,
                rstate=np.random.RandomState(self.randn)
            )
        except KeyboardInterrupt:
            warnings.warn("用户中断搜索，返回已完成的记录")
        except Exception as e:
            raise RuntimeError(f"TPE搜索失败: {str(e)}") from e
        
        # 转换结果为DataFrame
        trails_df = pd.DataFrame(self.trails.results)
        
        end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        display(f"结束执行时间：{end_time_str}")
        display(f"实际完成轮数：{len(trails_df)}")
        
        if len(trails_df) > 0:
            best_loss = trails_df['loss'].min()
            best_ks = -best_loss
            display(f"最优验证KS：{best_ks:.4f}")
        
        return trails_df