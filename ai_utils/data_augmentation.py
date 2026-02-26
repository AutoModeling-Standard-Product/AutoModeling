"""
数据增强模块 - 风险建模数据预处理与样本平衡工具

提供重采样、样本加权、Focal Loss等数据增强功能，
支持机构级数据平衡和类别不平衡处理。

Core Principles:
    - Data Quality: 所有重采样必须保持数据分布的合理性
    - Institutional Awareness: 支持机构级别的样本平衡策略
    - Reproducibility: 所有随机操作支持固定随机种子

Typical Usage:
    >>> # 机构级重采样
    >>> X_resampled, y_resampled = resampling(
    ...     X_train, y_train, ini_index=0,
    ...     algo='SMOTE', params={'sampling_strategy': 0.5, 'random_state': 42},
    ...     org_balance=True
    ... )
    >>> 
    >>> # 样本加权
    >>> weights = re_weight_by_org(
    ...     y_train, tr_orgidx, scale=0.5,
    ...     broadcast_with_tar=True, balanced_badrate=0.15
    ... )

Author: Model Development Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm
from collections import Counter
from scipy import special, optimize

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, log_loss

# 尝试导入imbalanced-learn
try:
    from imblearn.over_sampling import (
        RandomOverSampler, ADASYN, SMOTE, SMOTENC, 
        BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
    )
    from imblearn.under_sampling import (
        RandomUnderSampler, TomekLinks, ClusterCentroids
    )
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    import warnings
    warnings.warn("imblearn未安装，重采样功能将不可用")


__all__ = [
    'resampling',
    'FocalLoss', 
    're_weight_by_org'
]


def resampling(
    X_tot: pd.DataFrame,
    y_tot: pd.DataFrame,
    ini_index: int,
    algo: str,
    params: Dict[str, Any],
    label: str = 'new_target',
    org_tag: str = 'new_org',
    org_balance: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """
    样本重采样 - 处理类别不平衡的数据增强方法
    
    基于imbalanced-learn库实现多种过采样和欠采样算法，
    支持机构级别的分层重采样，确保各机构样本分布均衡。
    
    Args:
        X_tot (pd.DataFrame): 训练集特征矩阵，必须包含org_tag列
        y_tot (pd.DataFrame): 训练集目标变量DataFrame（包含label和org_tag列）
        ini_index (int): 初始索引（保留参数，当前实现中未使用）
        algo (str): 重采样算法名称
            过采样选项: 'ADASYN', 'RandomOverSampler', 'SMOTE', 'SMOTENC',
                      'BorderlineSMOTE', 'KMeansSMOTE', 'SVMSMOTE'
            欠采样选项: 'RandomUnderSampler', 'TomekLinks', 'ClusterCentroids'
        params (Dict[str, Any]): 算法参数字典
            必需参数: 'sampling_strategy', 'random_state'
            可选参数: 'n_neighbors', 'k_neighbors', 'categorical_features',
                     'kind', 'n_jobs'（依算法而定）
        label (str): 目标变量列名，默认为'new_target'
        org_tag (str): 机构标识列名，默认为'new_org'
        org_balance (bool): 是否按机构分层重采样
            - True: 每个机构独立重采样后合并
            - False: 全量数据统一重采样
    
    Returns:
        Union[Tuple[np.ndarray, np.ndarray], None]:
            - (X_resampled, y_resampled): 重采样后的特征和标签数组
            - None: 重采样失败时返回None
    
    Raises:
        AssertionError: 当algo不在支持的算法列表中时
        ValueError: 当必要参数缺失或数据格式错误时
        RuntimeError: 重采样过程中出现异常
    
    Business Context:
        类别不平衡是风险建模的常见问题（坏样本通常<5%）：
        
        算法选择指南：
        ┌─────────────────┬─────────────────────────────────────────┐
        │ 算法             │ 适用场景                                 │
        ├─────────────────┼─────────────────────────────────────────┤
        │ RandomOverSampler│ 简单快速，无信息损失，但可能过拟合        │
        │ SMOTE           │ 生成合成样本，适合连续特征               │
        │ BorderlineSMOTE │ 关注分类边界，提升模型区分能力           │
        │ ADASYN          │ 自适应生成，对难分类样本生成更多样本     │
        │ RandomUnderSampler│ 快速减少样本量，可能丢失信息           │
        │ TomekLinks      │ 清理噪声样本，通常与过采样联合使用       │
        └─────────────────┴─────────────────────────────────────────┘
        
        机构平衡策略（org_balance=True）：
        - 解决不同机构样本量差异大的问题
        - 保持各机构内部正负样本比例一致
        - 避免大机构主导模型训练
        
        sampling_strategy常用设置：
        - 'auto': 自动平衡至1:1
        - float (0,1): 少数类/多数类的比例
        - dict: 各类别具体样本数量
        
        质量检查：
        - 重采样后检查各机构样本分布
        - 验证合成样本的合理性（无异常值）
        - 监控模型在原始分布上的泛化能力
    
    Example:
        >>> # 基础SMOTE过采样
        >>> params = {
        ...     'sampling_strategy': 0.3,  # 坏:好 = 0.3:1
        ...     'random_state': 42,
        ...     'k_neighbors': 5
        ... }
        >>> X_res, y_res = resampling(
        ...     X_train, y_train, ini_index=0,
        ...     algo='SMOTE', params=params
        ... )
        >>> print(f"重采样后: {len(X_res)} 样本")
        
        >>> # 机构级分层重采样
        >>> X_res, y_res = resampling(
        ...     X_train, y_train, ini_index=0,
        ...     algo='BorderlineSMOTE', params=params,
        ...     org_balance=True
        ... )
    
    Note:
        - 使用imbalanced-learn库，需提前安装
        - 缺失值必须在调用前处理（NaN会导致欠采样报错）
        - 大数据集建议使用RandomUnderSampler或分块处理
        - 重采样后建议验证特征分布是否保持一致
    """
    # 检查imbalanced-learn是否可用
    if not IMBLEARN_AVAILABLE:
        raise RuntimeError("imblearn库未安装，无法执行重采样")
    
    # 内部函数：选择并执行重采样器
    def resampler_selection(
        X: pd.DataFrame,
        y: pd.Series,
        algo: str,
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """根据算法名称创建并执行重采样器"""
        # 验证算法名称
        supported_algos = [
            'ADASYN', 'RandomOverSampler', 'SMOTE', 'SMOTENC',
            'BorderlineSMOTE', 'KMeansSMOTE', 'SVMSMOTE',
            'RandomUnderSampler', 'TomekLinks', 'ClusterCentroids'
        ]
        assert algo in supported_algos, f"不支持的算法: {algo}"
        
        # 创建重采样器实例
        if algo == 'ADASYN':
            sampler = ADASYN(
                sampling_strategy=params['sampling_strategy'],
                random_state=params['random_state'],
                n_neighbors=params.get('n_neighbors', 5)
            )
        elif algo == 'RandomOverSampler':
            sampler = RandomOverSampler(
                sampling_strategy=params['sampling_strategy'],
                random_state=params['random_state']
            )
        elif algo == 'SMOTE':
            sampler = SMOTE(
                sampling_strategy=params['sampling_strategy'],
                random_state=params['random_state'],
                k_neighbors=params.get('k_neighbors', 5)
            )
        elif algo == 'SMOTENC':
            sampler = SMOTENC(
                categorical_features=params['categorical_features'],
                random_state=params['random_state']
            )
        elif algo == 'BorderlineSMOTE':
            sampler = BorderlineSMOTE(
                sampling_strategy=params['sampling_strategy'],
                random_state=params['random_state'],
                kind=params.get('kind', 'borderline-1')
            )
        elif algo == 'KMeansSMOTE':
            sampler = KMeansSMOTE(
                sampling_strategy=params['sampling_strategy'],
                random_state=params['random_state']
            )
        elif algo == 'SVMSMOTE':
            sampler = SVMSMOTE(
                sampling_strategy=params['sampling_strategy'],
                random_state=params['random_state']
            )
        elif algo == 'RandomUnderSampler':
            sampler = RandomUnderSampler(
                sampling_strategy=params['sampling_strategy'],
                random_state=params['random_state']
            )
        elif algo == 'TomekLinks':
            sampler = TomekLinks(
                sampling_strategy=params.get('sampling_strategy', 'auto'),
                n_jobs=params.get('n_jobs', -1)
            )
        elif algo == 'ClusterCentroids':
            sampler = ClusterCentroids(
                sampling_strategy=params['sampling_strategy'],
                random_state=params['random_state']
            )
        
        # 执行重采样
        return sampler.fit_resample(X, y)
    
    try:
        # 执行重采样逻辑
        if not org_balance:
            # 全量统一重采样
            return resampler_selection(
                X_tot.drop(columns=[org_tag]),
                y_tot[label],
                algo,
                params
            )
        else:
            # 机构级分层重采样
            if org_tag not in X_tot.columns or org_tag not in y_tot.columns:
                raise ValueError(f"X或y中缺少机构标识列'{org_tag}'")
            
            result_X, result_y = [], []
            
            for org_name in tqdm(X_tot[org_tag].unique(), desc="机构级重采样"):
                # 筛选当前机构数据
                X_org = X_tot[X_tot[org_tag] == org_name].drop(columns=[org_tag])
                y_org = y_tot[y_tot[org_tag] == org_name][label]
                
                # 执行重采样
                X_resampled, y_resampled = resampler_selection(
                    X_org, y_org, algo, params
                )
                
                result_X.append(X_resampled)
                result_y.append(y_resampled)
            
            # 合并所有机构结果
            return np.vstack(result_X), np.concatenate(result_y)
            
    except Exception as e:
        raise RuntimeError(f"重采样失败: {str(e)}")


class FocalLoss:
    """
    Focal Loss损失函数 - 处理类别不平衡的焦点损失实现
    
    Focal Loss通过降低易分类样本的权重，使模型更关注难分类的样本，
    特别适用于类别极度不平衡的风险建模场景。
    
    数学公式:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        其中:
        - p_t: 真实类别的预测概率
        - α_t: 类别权重（平衡正负样本）
        - γ: 聚焦参数（降低易分类样本权重）
    
    Attributes:
        gamma (float): 聚焦参数，γ>0时降低易分类样本权重
        alpha (Optional[float]): 正负样本权重，None表示1:1，float表示正样本权重
    
    Business Context:
        Focal Loss在风险建模中的应用：
        
        相比传统交叉熵损失的优势：
        ┌─────────────────┬──────────────────────────────────────────┐
        │ 场景             │ Focal Loss效果                           │
        ├─────────────────┼──────────────────────────────────────────┤
        │ 类别极度不平衡   │ 避免模型偏向预测多数类（好样本）          │
        │ 难分样本多       │ 自动提升难分样本的权重                    │
        │ 噪声标签         │ 降低易分类噪声样本的影响                  │
        │ 模型校准         │ 产生更可靠的预测概率                      │
        └─────────────────┴──────────────────────────────────────────┘
        
        超参数调优建议：
        - gamma (聚焦强度):
          * 1-2: 轻微聚焦，适合轻度不平衡
          * 2-3: 中等聚焦，通用选择
          * 3-5: 强聚焦，适合极度不平衡（坏样本<1%）
        
        - alpha (类别权重):
          * None: 不调整类别权重
          * 0.25-0.5: 增加正样本权重，平衡类别影响
          * 通常设置为反比于类别频率
        
        LightGBM集成:
        - 提供自定义目标函数（lgb_obj）
        - 提供评估函数（lgb_eval）
        - 支持XGBoost（xgb_eval）
    
    Example:
        >>> # 创建Focal Loss实例
        >>> focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
        >>> 
        >>> # LightGBM训练
        >>> model = lgb.train(
        ...     params,
        ...     train_set,
        ...     fobj=focal_loss.lgb_obj,
        ...     feval=focal_loss.lgb_eval
        ... )
        >>> 
        >>> # 直接计算损失
        >>> y_true = np.array([0, 1, 0, 1])
        >>> y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        >>> loss = focal_loss(y_true, y_pred)
        >>> print(f"Focal Loss: {loss.mean():.4f}")
    
    Note:
        - γ=0时退化为带权重的交叉熵损失
        - alpha=None时退化为普通Focal Loss
        - 需要scipy库支持优化初始化分数
        - 数值稳定性：内部使用clip限制概率范围[1e-15, 1-1e-15]
    
    References:
        Lin T Y, Goyal P, Girshick R, et al. Focal loss for dense object 
        detection. ICCV 2017.
    """

    def __init__(self, gamma: float, alpha: Optional[float] = None) -> None:
        """
        初始化Focal Loss
        
        Args:
            gamma (float): 聚焦参数，控制对易分类样本的降级程度
                - 推荐值：1.0-5.0
                - 越大越关注难分样本
            alpha (Optional[float]): 正样本权重，用于平衡类别
                - None: 不调整类别权重
                - float (0,1): 正样本权重，通常设为反比于类别比例
        
        Example:
            >>> # 极度不平衡场景（坏样本率<1%）
            >>> fl = FocalLoss(gamma=3.0, alpha=0.01)
            >>> 
            >>> # 轻度不平衡场景
            >>> fl = FocalLoss(gamma=1.5, alpha=0.3)
        """
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y: np.ndarray) -> np.ndarray:
        """
        计算类别权重α_t
        
        Args:
            y (np.ndarray): 真实标签（0或1）
        
        Returns:
            np.ndarray: 每个样本的权重
        """
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        计算真实类别的预测概率p_t
        
        Args:
            y (np.ndarray): 真实标签
            p (np.ndarray): 预测概率
        
        Returns:
            np.ndarray: p_t值（已裁剪避免数值不稳定）
        """
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        计算Focal Loss
        
        Args:
            y_true (np.ndarray): 真实标签
            y_pred (np.ndarray): 预测概率
        
        Returns:
            np.ndarray: 每个样本的Focal Loss值
        """
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        计算Focal Loss的一阶梯度（用于梯度提升）
        
        Args:
            y_true (np.ndarray): 真实标签
            y_pred (np.ndarray): 预测概率
        
        Returns:
            np.ndarray: 梯度值
        """
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        计算Focal Loss的二阶梯度（Hessian，用于梯度提升）
        
        Args:
            y_true (np.ndarray): 真实标签
            y_pred (np.ndarray): 预测概率
        
        Returns:
            np.ndarray: Hessian值
        """
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true: np.ndarray) -> float:
        """
        计算最优初始分数（用于梯度提升初始化）
        
        Args:
            y_true (np.ndarray): 真实标签
        
        Returns:
            float: 最优初始log-odds值
        """
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, preds: np.ndarray, train_data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        LightGBM自定义目标函数
        
        Args:
            preds (np.ndarray): 当前预测值（log-odds）
            train_data (lgb.Dataset): LightGBM数据集
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (梯度, Hessian)
        """
        y = train_data.get_label()
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, preds: np.ndarray, train_data: lgb.Dataset) -> Tuple[str, float, bool]:
        """
        LightGBM自定义评估函数
        
        Args:
            preds (np.ndarray): 当前预测值
            train_data (lgb.Dataset): LightGBM数据集
        
        Returns:
            Tuple[str, float, bool]: (评估名, 评估值, 是否越大越好)
        """
        y = train_data.get_label()
        p = special.expit(preds)
        return 'focal_loss', self(y, p).mean(), False

    def xgb_eval(self, preds: np.ndarray, train_data: Any) -> Tuple[str, float]:
        """
        XGBoost自定义评估函数
        
        Args:
            preds (np.ndarray): 当前预测值
            train_data: XGBoost数据集
        
        Returns:
            Tuple[str, float]: (评估名, 评估值)
        """
        y = train_data.get_label()
        p = special.expit(preds)
        return 'focal_loss', self(y, p).mean()


def re_weight_by_org(
    y_tr: pd.Series,
    tr_orgidx: Dict[str, List[int]],
    scale: float = 1.0,
    broadcast_with_tar: bool = False,
    balanced_badrate: Optional[float] = 0.15
) -> pd.Series:
    """
    按机构重新加权样本 - 处理机构间样本不平衡的权重计算
    
    根据机构信息和目标变量计算样本权重，支持多种加权策略：
    1. 基于自定义坏样率的平衡（balanced_badrate）
    2. 基于目标变量的分层加权（broadcast_with_tar）
    3. 简单的机构样本量平衡
    
    Args:
        y_tr (pd.Series): 训练集目标变量，索引对应样本
        tr_orgidx (Dict[str, List[int]]): 机构索引字典
            - key: 机构名称
            - value: 该机构样本的索引列表
        scale (float): 权重缩放级别，0为无缩放，1为线性缩放
        broadcast_with_tar (bool): 是否基于目标变量分层加权
            - True: 同时考虑机构和目标变量分布
            - False: 仅考虑机构样本量
        balanced_badrate (Optional[float]): 目标坏样率
            - None: 不使用坏样率平衡
            - float (0,1): 各机构目标坏样率
    
    Returns:
        pd.Series: 样本权重序列，索引与y_tr对齐
    
    Raises:
        ValueError: 当参数设置冲突或数据格式错误时
        RuntimeError: 权重计算过程中出现异常
    
    Business Context:
        机构级加权是风险建模的关键数据预处理步骤：
        
        应用场景：
        1. 机构样本量差异大：防止大机构主导模型
        2. 机构坏样率差异大：平衡不同风险水平的机构
        3. 联合建模：多机构数据合并训练时的样本平衡
        
        加权策略对比：
        ┌──────────────────┬──────────────────────────────────────────┐
        │ 策略              │ 适用场景                                 │
        ├──────────────────┼──────────────────────────────────────────┤
        │ balanced_badrate │ 需要统一各机构坏样率到目标值             │
        │ broadcast_with_tar│ 同时考虑机构分布和目标变量分布          │
        │ 仅scale          │ 简单平衡机构样本量                       │
        └──────────────────┴──────────────────────────────────────────┘
        
        balanced_badrate计算逻辑：
        - 计算各机构正负样本数
        - 根据目标坏样率计算正负样本权重
        - 同时考虑机构间样本量平衡（beta因子）
        - 最终权重 = alpha * beta（对数压缩后）
        
        broadcast_with_tar计算逻辑：
        - 构建机构×目标的样本数矩阵
        - 按最大样本数计算缩放因子
        - 应用scale参数控制缩放强度
        - wgt = (max_count / actual_count)^scale
        
        质量检查：
        - 加权后检查各机构有效样本量
        - 验证目标变量分布是否符合预期
        - 监控模型在各机构的性能差异
    
    Example:
        >>> # 基于目标坏样率的加权
        >>> weights = re_weight_by_org(
        ...     y_train, tr_orgidx,
        ...     balanced_badrate=0.15,  # 目标15%坏样率
        ...     scale=0.5
        ... )
        >>> print(f"权重范围: {weights.min():.2f} - {weights.max():.2f}")
        
        >>> # 分层加权（同时考虑机构和目标）
        >>> weights = re_weight_by_org(
        ...     y_train, tr_orgidx,
        ...     broadcast_with_tar=True,
        ...     scale=0.8
        ... )
    
    Note:
        - 优先级: balanced_badrate > broadcast_with_tar > scale
        - 权重经过对数压缩（log1p）防止极端值
        - 返回的Series索引与输入y_tr对齐
        - 内部使用pandas进行机构分组计算
    """
    
    def index_dict_2_series(d: Dict[str, List[int]]) -> pd.Series:
        """
        将索引字典转换为Series
        
        Args:
            d: 机构索引字典
        
        Returns:
            pd.Series: 机构标签Series，索引为样本索引
        """
        values = []
        indices = []
        for key, index_list in d.items():
            values.extend([key] * len(index_list))
            indices.extend(index_list)
        return pd.Series(values, index=indices)
    
    try:
        # 将orgid转为Series以便与y_tr对接
        sr_org = index_dict_2_series(tr_orgidx)
        
        # 策略1: 基于自定义坏样率的平衡
        if balanced_badrate is not None:
            # 构建临时DataFrame
            df_temp = pd.DataFrame({
                'tar': y_tr,
                'org': sr_org
            })
            
            # 计算机构×目标的样本数矩阵
            df_group = df_temp.pivot_table(
                index='tar',
                columns='org',
                aggfunc='size',
                fill_value=0
            )
            
            # 计算alpha（坏样率平衡因子）
            # w[1] = max_count * badrate / ((1-badrate) * min_count)
            w = pd.DataFrame(
                df_group.max(axis=0) * balanced_badrate / 
                ((1 - balanced_badrate) * df_group.min(axis=0)),
                columns=['1']
            )
            
            # 计算beta（机构样本量平衡因子）
            w['0'] = df_group.sum(axis=0).max() / df_group.sum(axis=0)
            
            # 合并并应用对数压缩
            w['1'] = w['1'] * w['0']
            w['0'] = np.log1p(w['0'])
            w['1'] = np.log1p(w['1'])
            
            # 转换为长格式
            w = w.pivot_table(columns=['org'])
            w['org'] = [0.0, 1.0]
            
            # 应用权重到每个样本
            res_ = df_temp.groupby(['tar', 'org']).apply(
                lambda x: pd.DataFrame({
                    'tar': x['tar'].iloc[0],
                    'org': x['org'].iloc[0],
                    'wgt': w[w.org == x['tar'].iloc[0]][x['org'].iloc[0]]
                })
            ).reset_index(drop=True)
            
            # 合并回原始索引
            res = pd.merge(df_temp, res_, on=['tar', 'org'], how='left')
            res = res.set_index(df_temp.index)
            
            return res['wgt']
        
        # 策略2: 基于目标变量的分层加权
        elif broadcast_with_tar:
            df_temp = pd.DataFrame({
                'tar': y_tr,
                'org': sr_org
            })
            
            # 计算机构×目标的样本数
            df_group = df_temp.pivot_table(
                index='tar',
                columns='org',
                aggfunc='size',
                fill_value=0
            )
            
            # 计算最大样本数
            max_val = df_group.max().max()
            
            # 计算缩放因子
            df_group = (max_val / df_group) ** scale
            df_group = df_group.stack().to_dict()
            
            # 应用权重映射
            value_map = lambda row: df_group.get((row['tar'], row['org']), 1.0)
            df_temp['wgt'] = df_temp.apply(value_map, axis=1)
            
            return df_temp['wgt']
        
        # 策略3: 简单的机构样本量平衡
        else:
            # 统计各机构样本数
            sr_temp = sr_org.value_counts()
            max_val = sr_temp.max()
            
            # 计算缩放因子
            sr_temp = (max_val / sr_temp) ** scale
            sr_temp = sr_temp.to_dict()
            
            # 映射到索引
            return index_dict_2_series({
                sr_temp[key]: tr_orgidx[key]
                for key in tr_orgidx.keys()
            })
            
    except Exception as e:
        raise RuntimeError(f"权重计算失败: {str(e)}")

