import numpy
import pandas as pd
import pickle
from tqdm.auto import tqdm
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, log_loss

try:
    from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, SMOTENC, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids
except:
    from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, SMOTENC, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids

from scipy import special, optimize
from scipy.misc import derivative


def over_sampling(
    sampler，
    X_tr: pd.DataFrame,
    y_tr: pd.DataFrame,
    tr_orgidx: dict,
    start_index: int,
    fill_value: int = -999
):
    """
    :param sampler       选择采样器，需要一次性定义算法及参数，传入范例：SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
    :param X_tr          切分后的训练集特征
    :param y_tr          切分后的训练集目标变量
    :param tr_orgidx     训练集样本对应的机构id
    :param start_index   训练集重新生成索引的起始点，建议设置为当前整体dataframe索引最大值+1
    :param fill_value    imblearn大部分重采样算法均需要预先填充缺失值，以保证k临近搜索可以正确执行

    返回过采样之后的X_tr，y_tr，tr_orgidx
    """
    
    try:
        result_X, result_y = [], []
        result_index = {org: [] for org in tr_orgidx.keys()}
        start_index_ = start_index
        for org_name in tqdm(tr_orgidx.keys()):
            result_X_org, result_y_org = sampler.fit_resample(
                X_tr.loc[tr_orgidx[org]].fillna(fill_value), # 这里需要做缺失值填充
                y_tr.loc[tr_orgidx[org]]
            )
            result_index_org = np.arange(start_index_, start_index_ + len(result_X_org))
            result_X_org = pd.DataFrame(result_X_org, columns=X_tr.columns, index=result_index_org)
            result_y_org = pd.Series(result_y_org, index=result_index_org)

            result_X.append(result_X_org)
            result_y.append(result_y_org)
            result_index[org] = result_index_org
            start_index_ += len(result_X_org)
            
        return pd.concat(result_X).replace(fill_value, np.nan), pd.concat(result_y), result_index
    except Exception as e:
        display(f'过采样失败：{e}')
        return


def re_weight_by_org(y_tr: pd.Series, tr_orgidx: dict, scale: float = 1.0, broadcast_with_tar: bool = False):
    """
    :param y_tr                 训练集目标变量
    :param tr_orgidx            机构标识：用于区分样本权重
    :param scale                缩放级别：格式为float，0.0为无缩放，1.0为线性缩放
    :param broadcast_with_tar   是否基于目标变量划分权重，True代表同时使用机构标识和目标变量进行二维划分
    """
    def index_dict_2_series(d: dict):
        values = []
        indices = []
        for key, index_list in d.items():
            values.extend([key] * len(index_list))
            indices.extend(index_list)
        return pd.Series(values, index=indices)
    # 将orgid转为Series以便与y_tr拼接
    sr_org = index_dict_2_series(tr_orgidx)
    if broadcast_with_tar:
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
