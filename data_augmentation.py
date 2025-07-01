import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, log_loss

# imblearn和TensorFlow可能会存在导入顺序的问题，解决方法为二次导入
try:
    from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, SMOTENC, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids
except:
    from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, SMOTENC, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids

from scipy import special, optimize
from scipy.misc import derivative


def resampling(
    X_tot: pd.DataFrame, 
    y_tot: pd.DataFrame,
    ini_index: int,
    algo: str, 
    params: dict, 
    label: str = 'new_target',
    org_tag: str = 'new_org',
    org_balance: bool = False
):
    """
    :param X_tot          切分后的训练集自变量特征。需要包含org_tag
    :param y_tot          切分后的训练集因变量。注意，这里的y_tot是DataFrame而非Series，因为包含了label和org_tag两部分
    :param algo           采样器类型
    :param params         重采样算法对应的参数字典，一般的，sampling_strategy和random_state是必选指定项
    :param label          因变量标识
    :param org_tag        机构名称标识，这一列必须存在于特征矩阵中
    :param org_balance    是否对每家机构都单独重采样，保持机构间坏样率均衡
    """
    # 重采样，这里不在函数名称上区分过采样或欠采样，由具体的算法与参数来决定。
    # 注意！这里假设缺失值已经进行了填充。nan存在会导致欠采样报错。
    def resampler_selection(X: pd.DataFrame, y: pd.DataFrame, algo: str, params: dict):
        try:
            assert algo in [
                'ADASYN', 'RandomOverSampler', 'SMOTE', 'SMOTENC', 'BorderlineSMOTE', 'KMeansSMOTE', 'SVMSMOTE',    
                'RandomUnderSampler', 'TomekLinks', 'ClusterCentroids'                                              
            ]
            if algo == 'ADASYN': 
                sampler = ADASYN(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'], n_neighbors=params['n_neighbors'])
            elif algo == 'RandomOverSampler': 
                sampler = RandomOverSampler(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
            elif algo == 'SMOTE':  
                sampler = SMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'], k_neighbors=params['k_neighbors'])
            elif algo == 'SMOTENC':  
                sampler = SMOTENC(categorical_features=params['categorical_features'], random_state=params['random_state'])
            elif algo == 'BorderlineSMOTE':  
                sampler = BorderlineSMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'], kind=params['kind'])
            elif algo == 'KMeansSMOTE':  
                sampler = KMeansSMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
            elif algo == 'SVMSMOTE':  
                sampler = SVMSMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
            elif algo == 'RandomUnderSampler': 
                sampler = RandomUnderSampler(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
            elif algo == 'TomekLinks': 
                sampler = TomekLinks(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'], n_jobs=params['n_jobs'])
            elif algo == 'ClusterCentroids':  
                sampler = ClusterCentroids(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
            
            return sampler.fit_resample(X, y)
        
        except Exception as e:
            display(f'重采样失败：{e}')
            
    if not org_balance:
        return resampler_selection(X_tot.drop(columns=[org_tag]), y_tot[label], algo, params)
    else:
        if org_tag in X_tot.columns and org_tag in y_tot.columns:
            result_X, result_y = [], []
            for org_name in tqdm(X_tot[org_tag].unique()):
                X_org = X_tot[X_tot[org_tag] == org_name].drop(columns=[org_tag])
                y_org = y_tot[y_tot[org_tag] == org_name].drop(columns=[org_tag])
                X_resampled, y_resampled = resampler_selection(X_org, y_org, algo, params)
                result_X.append(X_resampled)
                result_y.append(y_resampled)
            return np.vstack(result_X), np.concatenate(result_y)
        else:
            display(f'X或y的机构名称缺失，无法进行分机构重采样。')
            return 


class FocalLoss:

    def __init__(self, gamma, alpha=None):
        # 使用FocalLoss只需要设定以上两个参数,如果alpha=None,默认取值为1
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        # alpha 参数, 根据FL的定义函数,正样本权重为self.alpha,负样本权重为1 - self.alpha
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        # pt和p的关系
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better
    
    def xgb_eval(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        return 'focal_loss', self(y, p).mean()
    
    
    
def re_weight_by_org(y_tr: pd.Series, tr_orgidx: dict, scale: float = 1.0, broadcast_with_tar: bool = False, balanced_badrate: float=0.15):
    """
    :param y_tr                   训练集目标变量
    :param tr_orgidx              机构标识，用于区分权重
    :param scale                  缩放级别，格式为float，0为无缩放，1为线性缩放
    :param broadcast_with_tar     是否基于目标变量分层加权
    :param balanced_badrate       机构内坏样率
    
    返回一个权重序列，索引与训练集的X和y相对应。
    """
    
    def index_dict_2_series(d):
        values = []
        indices = []
        for key, index_list in d.items():
            values.extend([key] * len(index_list))
            indices.extend(index_list)
        return pd.Series(values, index=indices)
    
    # 将orgid转为Series以便与y_tr对接
    sr_org = index_dict_2_series(tr_orgidx)
    
    ## 根据自定义的坏样率平衡，首先对每个机构内平衡得到alpha。然后根据机构之间的样本数平衡得到beta, 最后的权重平衡参数为alpha * beta,
    ## 这样可以将坏样率也作为超参数寻优
    if balanced_badrate is not None:
        df_temp = pd.DataFrame({
            'tar': y_tr, 
            'org': sr_org
        })
        
        df_group = df_temp.pivot_table(index='tar', columns='org', aggfunc='size', fill_value=0)
        w = pd.DataFrame(df_group.max(axis=0) * balanced_badrate / ((1-balanced_badrate) * df_group.min(axis=0)), columns=['1'])
        w['0'] = df_group.sum(axis=0).max() / df_group.sum(axis=0)
        w['1'] = w['1']*w['0']
        #w['0'] = 1
        w = w.pivot_table(columns=['org'])
        w['org'] = [0.0, 1.0]
        #w = np.log1p(w)
        res_ = df_temp.groupby(['tar', 'org']).apply(lambda x: pd.DataFrame({
            'tar': x['tar'].iloc[0],
            'org': x['org'].iloc[0],
            'wgt': w[w.org==x['tar'].iloc[0]][x['org'].iloc[0]]
        })).reset_index(drop=True)
        res = pd.merge(df_temp, res_, on=['tar', 'org'], how='left')
        res = res.set_index(df_temp.index)
        
        return res['wgt']
    
    elif broadcast_with_tar: 
        df_temp = pd.DataFrame({
            'tar': y_tr, 
            'org': sr_org
        })
        df_group = df_temp.pivot_table(index='tar', columns='org', aggfunc='size', fill_value=0)
        max_val = df_group.max().max()
        df_group = (max_val / df_group) ** scale
        df_group = df_group.stack().to_dict()
        
        value_map = lambda row: df_group.get((row['tar'], row['org']), None)
        df_temp['wgt'] = df_temp.apply(value_map, axis=1)
        return df_temp['wgt']
    else:
        sr_temp = sr_org.value_counts()
        max_val = sr_temp.max()
        sr_temp = (max_val / sr_temp) ** scale
        sr_temp = sr_temp.to_dict()
        return index_dict_2_series({sr_temp[key]: tr_orgidx[key] for key in tr_orgidx.keys()})