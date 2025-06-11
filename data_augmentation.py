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


def resampling(
    X_tot: pd.DataFrame,
    y_tot: pd.DataFrame,
    algo: str,
    params: dict,
    label: str = 'new_target',
    org_tag: str = 'new_org',
    org_balance: bool = False
):
    """
    :param X_tot          切分后的训练集自变量特征，需要包含org_tag
    :param y_tot          切分后的训练集因变量。注意，由于需要包含label和org_tag两部分，这里是DataFrame而非Series
    :param algo           选择采样器类型
    :param params         采样器对应参数字典，一般的，sampling_strategy和random_state是必选指定项
    :param label          因变量标识
    :param org_tag。      机构名称标识，必须存在于X_tot和y_tot中以便采样
    :param org_balance    是否对每家机构单独重采样，以保持机构间坏样率的一致性
    """

    def resampler_selection(X: pd.DataFrame, y: pd.DataFrame, algo: str, params: dict):
        try:
            assert algo in [
                'RandomOverSampler', 'ADASYN', 'SMOTE', 'SMOTENC', 'BorderlineSMOTE', 'KMeansSMOTE', 'SVMSMOTE',
                'RandomUnderSampler', 'TomekLinks', 'ClusterCentroids'
            ]
            if algo == 'ADASYN':
                sampler = ADASYN(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'], n_neighbors=['n_neighbors'])
            elif algo == 'RandomOverSampler':
                sampler = RandomOverSampler(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
            elif algo == 'SMOTE':
                sampler = SMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])          
            elif algo == 'SMOTENC':
                sampler = SMOTENC(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
            elif algo == 'BorderlineSMOTE':
                sampler = BorderlineSMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
            elif algo == 'KMeansSMOTE':
                sampler = KMeansSMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])            
            elif algo == 'SVMSMOTE':
                sampler = SVMSMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
            elif algo == 'RandomUnderSampler':
                sampler = RandomUnderSampler(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
            elif algo == 'TomekLinks':
                sampler = TomekLinks(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
            elif algo == 'ClusterCentroids':
                sampler = ClusterCentroids(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
            
            return sampler.fit_resample(X, y)

        except Exception as e:
            display(f'采样器运作失败：{e}')

    try:
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
                return np.vstack(X_resampled), np.concatenate(result_y)
            else:
                display('X或y的机构名称缺失，无法分机构重采样')
                return
    except Exception as e:
            display(f'重采样失败：{e}')