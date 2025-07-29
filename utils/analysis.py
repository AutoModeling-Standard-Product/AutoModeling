from utils.func import *

def drop_abnormal_ym(**kwargs)-> pd.DataFrame:
    data = kwargs.get("data").copy()
    minYmBadsample = kwargs.get("minYmBadsample")
    minYmSample = kwargs.get("minYmSample")
    assert all(v in data.columns for v in ['new_org', 'new_date', 'new_date_ym', 'new_target']), "输入的数据列不符合命名要求"
    datasetStatis = org_analysis(data)
    datasetStatis = datasetStatis[(datasetStatis.单月坏样本数<minYmBadsample) | (datasetStatis.单月总样本数<minYmSample)]
    condition_pre = datasetStatis.apply(lambda row: (row.机构==data['new_org']) & (row.年月==data['new_date_ym']), axis=1)
    condition = condition_pre.any(axis=0)
    display(f"输入data样本数为{data.shape[0]}, 限制每个机构每月坏样本数>={minYmBadsample}, 样本数>={minYmSample}")
    data = data[~condition]
    display(f"删去异常月份后数据大小为{data.shape[0]}")
    return data

def drop_highmiss_features(**kwargs)-> pd.DataFrame:
    data = kwargs.get("data").copy()
    miss_org = kwargs.get("miss_org")
    miss_channel = kwargs.get('miss_channel')
    ratio = kwargs.get("ratio")
    cnt = kwargs.get("cnt")
    miss_org['是高缺失机构'] = np.where(miss_org['总缺失率']>ratio, 1, 0)
    miss_org['高缺失机构个数'] = miss_org.groupby('变量')['是高缺失机构'].transform('sum')
    miss_org = miss_org[miss_org.高缺失机构个数>cnt]
    miss_channel = miss_channel[miss_channel.总缺失率>ratio]
    display(f"输入data列数为{data.shape[1]}")
    deleted_feas = list(set(list(miss_channel.变量) + list(miss_org.变量)).intersection(set(data.columns)))
    data.drop(columns=deleted_feas, inplace=True)
    if len(deleted_feas)>0:
        display(f"删去高缺失变量后data列数为{data.shape[1]}, 删去{len(deleted_feas)}列, 删除标准：1.变量在多个机构下不满足单机构缺失率条件, 2.变量在渠道或总体下不满足缺失率条件")
    else:
        display('全部变量都不满足删除条件')
    return data

def drop_lowiv_features(**kwargs) -> pd.DataFrame:
    data = kwargs.get('data').copy()
    res_iv_org = kwargs.get('res_iv_org')
    res_iv_channel = kwargs.get('res_iv_channel')
    miniv_org = kwargs.get('miniv_org')
    miniv_channel = kwargs.get("miniv_channel")
    cnt = kwargs.get('cnt')
    res_iv_org['是低iv机构'] = np.where(res_iv_org['iv']<miniv_org, 1, 0)
    res_iv_org['低iv机构数'] = res_iv_org.groupby('变量')['是低iv机构'].transform('sum')
    drop_features = res_iv_org[res_iv_org.低iv机构数>cnt]['变量']
    keep_features = res_iv_channel[res_iv_channel.iv>=miniv_channel]['变量']
    drop_features = list(set((set(drop_features)-set(keep_features))).intersection(data.columns))
    if len(drop_features)>0:
        display(f"去除在{cnt}上个机构iv都小于{miniv_org} 且在任何渠道上iv都小于{miniv_channel}的变量, 共{len(drop_features)}个")
        data.drop(columns=drop_features, inplace=True)
    else:
        display("没有符合删去条件的变量")
    return data

def drop_highpsi_features(**kwargs)-> pd.DataFrame:
    data = kwargs.get("data").copy()
    res_psi_org = kwargs.get("res_psi_org")
    ratio = kwargs.get('ratio')
    cnt = kwargs.get("cnt")
    res_psi_org = res_psi_org[['机构', '变量', '最大psi']].drop_duplicates(subset=['机构', '变量'])
    res_psi_org = res_psi_org[res_psi_org.最大psi>ratio]
    res_psi_org['高psi机构数'] = res_psi_org.groupby('变量')['机构'].transform('count')
    drop_features = list(res_psi_org[res_psi_org['高psi机构数'] > cnt]['变量'])
    drop_features = list(set(drop_features).intersection(set(data.columns)))
    if len(drop_features)>0:
        display(f"存在变量{len(drop_features)}个, 在{cnt}以上机构数出现高psi被删除")
        data.drop(columns=drop_features, inplace=True)
    else:
        display("没有变量符合删去要求")
    return data

def drop_highcorrelation_features(**kwargs) -> pd.DataFrame:
    data = kwargs.get('data').copy()
    res_iv_channel = kwargs.get('res_iv_channel')
    indices = kwargs.get('indices')
    channel = kwargs.get('channel')
    display(f'根据{channel}下iv值删去变量, 遍历高相似变量对直到变量对全为空, 对于iv>=0.01的变量对进行了特殊处理从而尽可能保留高价值变量')
    features = []
    drop_features = []
    
    for idx in np.arange(indices.shape[0]):
        features.append(indices.iloc[idx][0])
        features.append(indices.iloc[idx][1])
    features = list(set(features))
    
    while indices.shape[0]>0:
        res = res_iv_channel[(res_iv_channel.变量.isin(features))&(res_iv_channel.渠道==channel)]
        drop_ = list(res[res.iv==np.min(res.iv)]['变量'])
        ## 当要去除高iv变量时，高相似性变量对更新为删去drop_和与drop_对应的变量, 减少高iv变量的删除
        keep_feature = None
        if np.min(res.iv) >= 0.01:
            if len(drop_) > 1:
                drop_ = [drop_[0]]
            indices_ = indices[(indices['0'].isin(drop_)) | (indices['1'].isin(drop_))]
            keep_feature = list(set(list(indices_['0'])+list(indices_['1'])+drop_))
            indices = indices[~((indices['0'].isin(keep_feature))|(indices['1'].isin(keep_feature)))]
            features = [v for v in features if v not in keep_feature]
        else:
            indices = indices[~((indices['0'].isin(drop_))|(indices['1'].isin(drop_)))]
            features = [v for v in features if v not in drop_]
        drop_features += drop_
#         print(f"fea: {drop_}, fea1:{keep_feature}, iv:{np.min(res.iv)}")
#         print(f"indices shape: {indices.shape}")

    drop_features = list(set(drop_features).intersection(set(data.columns)))
    display(f'共删去{len(drop_features)}个变量')
    data.drop(columns=drop_features, inplace=True)
    return data

def drop_highnoise_features(**kwargs)-> Tuple[pd.DataFrame, pd.DataFrame]:
    data = kwargs.get('data').copy()
    assert all(v in data.columns for v in ['new_org', 'new_date', 'new_date_ym', 'new_target']), "输入的数据列不符合命名要求"
    max_depth = kwargs.get('max_depth')
    n_estimators = kwargs.get('n_estimators')
    features = list(set([v for v in data.columns if data[v].dtype!='O'])-set(['new_target']))
    data.replace({-1111:np.nan, -999:np.nan, -1:np.nan}, inplace=True)
    data = data[data.new_target.isin([0, 1])]
    X = data[features]
    Y = data['new_target']
    Y_permuted = np.random.permutation(Y)
    for i in range(20):
        Y_permuted = np.random.permutation(Y_permuted)
    res_df_ = pd.DataFrame()
    res_df_permuted_ = pd.DataFrame()
    for idx in np.arange(0, 2):
        random_n = np.random.randint(30)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=random_n)
        X_train_, X_test_, y_train_, y_test_ = train_test_split(X, Y_permuted, test_size=0.3, random_state=random_n)
        clf = get_fixed_lgb(max_depth, n_estimators)
        clf_ = get_fixed_lgb(max_depth, n_estimators)
        clf.fit(X_train,y_train)
        clf_.fit(X_train_,y_train_)
        train = clf.predict_proba(X_train)[:, 1]
        test = clf.predict_proba(X_test)[:, 1]
        train_ = clf_.predict_proba(X_train_)[:, 1]
        test_ = clf_.predict_proba(X_test_)[:, 1]
        auc_train = round(roc_auc_score(y_train, train), 3)
        auc_test = round(roc_auc_score(y_test, test), 3)
        auc_train_ = round(roc_auc_score(y_train_, train_), 3)
        auc_test_ = round(roc_auc_score(y_test_, test_), 3)
        res_df = pd.DataFrame({'name':clf.booster_.feature_name(),
                               'gain':clf.booster_.feature_importance(importance_type='gain')})
        res_df_permuted = pd.DataFrame({'name':clf_.booster_.feature_name(),
                               'gain':clf_.booster_.feature_importance(importance_type='gain')})
        res_df = res_df[res_df.gain>0]
        res_df = res_df.sort_values(by='gain', ascending=False)
        res_df_ = pd.concat([res_df_, res_df], axis=0) 
        res_df_permuted = res_df_permuted[res_df_permuted.gain>0]
        res_df_permuted = res_df_permuted.sort_values(by='gain', ascending=False)
        res_df_permuted_ = pd.concat([res_df_permuted_, res_df_permuted], axis=0)
        print("(AUC) train:{}, test:{}, train_permuted:{}, test_permuted:{}".format(auc_train, auc_test, auc_train_, auc_test_))
    res = res_df_.groupby('name').apply(lambda x: pd.Series({
        'name':x['name'].iloc[0],
        'gain': round(np.mean(x['gain']), 0)
    }))
    res = pd.DataFrame(res).drop_duplicates(subset=['name'])
    res_permuted = res_df_permuted_.groupby('name').apply(lambda x: pd.Series({
        'name':x['name'].iloc[0],
        'gain': round(np.mean(x['gain']), 0)
    }))
    res_permuted = pd.DataFrame(res_permuted).drop_duplicates(subset=['name'])
    res = res.sort_values(by=['gain'], ascending=False).reset_index(drop=True)
    res_permuted = res_permuted.sort_values(by=['gain'], ascending=False).reset_index(drop=True)
    res_ = pd.merge(res, res_permuted, on=['name'], how='left')
    res_.columns = ['name', 'gain', 'gain_permuted']
    drop_features = list(res_[abs(res_.gain-res_.gain_permuted)<50]['name'])
    if len(drop_features) > 0:
        display(f"共有{len(drop_features)}个变量存在过高噪音, 翻转y后建模gain值差异小于50")
        data.drop(columns=drop_features, inplace=True)
    else:
        display('没有符合条件的高噪音变量')
    return data, res_