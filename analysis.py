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
    deleted_feas = list(set(miss_channel.变量 + miss_org.变量).intersection(set(data.columns)))
    data.drop(columns=deleted_feas, inplace=True)
    if len(deleted_feas)>0:
        display(f"删去高缺失变量后data列数为{data.shape[1]}, 删去的变量是{deleted_feas}, 删除标准：1.变量在多个机构下不满足单机构缺失率条件, 2.变量在渠道或总体下不满足缺失率条件")
    else:
        display('全部变量都不满足删除条件')
    return data

def drop_lowiv_features(**kwargs) -> pd.DataFrame:
    data = kwargs.get('data').copy()
    res_iv_org = kwargs.get('res_iv_org')
    res_iv_channel = kwargs.get('res_iv_channel')
    miniv = kwargs.get('miniv')
    cnt = kwargs.get('cnt')
    
    res_iv_org['是低iv机构'] = np.where(res_iv_org['iv']<miniv, 1, 0)
    res_iv_org['低iv机构数'] = res_iv_org.groupby('变量')['是低iv机构'].transform('sum')
    drop_features = res_iv_org[res_iv_org.低iv机构数>cnt]['变量']
    keep_features = res_iv_channel[res_iv_channel.iv>=miniv]['变量']
    drop_features = list(set((set(drop_features)-set(keep_features))).intersection(data.columns))
    if len(drop_features)>0:
        display(f"去除在{cnt}上个机构iv都小于{miniv} 且在任何渠道上iv都小于{miniv}的变量, 共{len(drop_features)}个")
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
    if len(drop_features)>0:
        display(f"变量{drop_features}, 在{cnt}以上机构数出现高psi")
    else:
        display("没有变量符合删去要求")
    drop_features = list(set(drop_features).intersection(set(data.columns)))
    data.drop(columns=drop_features, inplace=True)
    return data

def dataset_analysis(**kwargs)->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  
    kwargs_ = kwargs
    data = kwargs.get('data').copy()
    bank_orgs = kwargs.get("bank_orgs")
    nonbank_orgs = kwargs.get("nonbank_orgs")
    
    display("----------------------------------")
    display('>>分｜不分机构统计坏样率')
    highmissing = missing_check(**kwargs)
    highmissing = highmissing[['机构', '变量', '单机构缺失率', '单机构坏样率', '总体缺失率', '总体坏样率']]
    
    display("----------------------------------")
    display('>>分机构｜渠道统计变量iv｜平均iv, 不分机构渠道统计变量iv')
    res_iv_0 = calculate_meaniv_channel(**kwargs)
    res_iv_1 = calculate_iv_traditionally(**kwargs)
    res_iv_1.rename(columns={'iv': '总体iv'}, inplace=True)
    
    display("----------------------------------")
    display(">>分渠道季度滑动统计变量iv, 分渠道逐月统计变量psi")
    display(">>银行")
    kwargs_['channel'] = '银行'
    data_ = data[data.new_org.isin(bank_orgs)]
    kwargs_.update({'data': data_})
    res_iv_20 = calculate_dynamiciv_channel(**kwargs_)
    res_psi_10 = calculate_psi_channel(**kwargs_)
    
    display(">>非银")
    kwargs_.update({'channel':'非银'})
    data_ = data[data.new_org.isin(nonbank_orgs)]
    kwargs_.update({'data': data_})
    res_iv_21 = calculate_dynamiciv_channel(**kwargs_)
    res_psi_11 = calculate_psi_channel(**kwargs_)
    
    display(">>三方")
    kwargs_.update({'channel':'三方'})
    data_ = data[~data.new_org.isin(nonbank_orgs+bank_orgs)]
    kwargs_.update({'data': data_})
    res_iv_22 = calculate_dynamiciv_channel(**kwargs_)
    res_psi_12 = calculate_psi_channel(**kwargs_)
    
    display(">>整体")
    kwargs_.update({'channel':'整体'})
    data_ = data.copy()
    kwargs_.update({'data': data_})
    res_iv_23 = calculate_dynamiciv_channel(**kwargs_)
    res_psi_13 = calculate_psi_channel(**kwargs_)
    
    res_iv_2 = pd.concat([res_iv_20, res_iv_21, res_iv_22, res_iv_23], axis=0)
    res_psi_1 = pd.concat([res_psi_10, res_psi_11, res_psi_12, res_psi_13], axis=0)
    
    display("----------------------------------")
    display(">>分｜不分机构逐月逐季度统计变量psi")
    res_psi_0 = calculate_psi_org(**kwargs)
    
    res_psi_2 = calculate_dypsi_org(**kwargs)
    
    data = kwargs.get('data')
    data['new_org'] = '整体'
    kwargs.update({'data': data})
    res_psi_1_ = calculate_psi_org(**kwargs)
    res_psi_1_.columns = ['渠道', '变量', '分渠道当前月份', '分渠道psi', '分渠道月份统计总数', '分渠道高psi月份计数', '渠道下最大psi']
    res_psi_1 = pd.concat([res_psi_1_, res_psi_1], axis=0)
    res_psi_2_ = calculate_dypsi_org(**kwargs)
    res_psi_2 = pd.concat([res_psi_2_, res_psi_2], axis=0)
    
    table1 = pd.merge(highmissing, res_iv_0[['渠道', '机构','变量','单机构iv', '渠道下平均iv', '渠道缺失率']], on=['机构', '变量'], how='right')
    table1 = table1.merge(res_iv_1, on=['变量'], how='left')
    table1 = pd.merge(res_psi_0[['机构', '变量', '单机构下最大psi']].drop_duplicates(subset=['机构', '变量']), table1, on=['机构', '变量'], how='right')
    table1 = pd.merge(res_psi_1[['渠道', '变量', '渠道下最大psi']].drop_duplicates(subset=['渠道', '变量']), table1, on=['渠道', '变量'], how='right')
    table1 = pd.merge(res_psi_2[['机构', '变量', '季度下最大psi']].drop_duplicates(subset=['机构', '变量']), table1, on=['机构', '变量'], how='right')
    table1 = table1[['渠道', '机构', '变量', '单机构iv', '渠道下平均iv', '总体iv', '单机构下最大psi', '渠道下最大psi', '季度下最大psi',
                     '单机构缺失率', '单机构坏样率', '总体缺失率', '总体坏样率', '渠道缺失率']]
    table1['渠道下平均iv'] = np.round(table1['渠道下平均iv'], 3)
    table1['总体iv'] = np.round(table1['总体iv'], 3)
    table1 = table1.sort_values(by=['渠道', '机构', '总体iv', '渠道下平均iv', '单机构iv'], ascending=False)
    
    table1_ = table1[['机构', '变量', '单机构iv', '单机构下最大psi', '单机构缺失率']].drop_duplicates(subset=['机构', '变量'])
    table1_['flag'] = np.where((table1_.单机构iv>0.02) & (table1_.单机构缺失率<0.95) & (table1_.单机构下最大psi<0.1), 1, 0)
    table1_ = table1_.groupby('机构').apply(lambda x: pd.Series({
        '机构':x['机构'].iloc[0],
        '单机构下变量合格率': np.round(np.mean(x['flag']), 4),
        '单机构下变量合格个数':np.sum(x['flag'])
    })).reset_index(drop=True)
    table1 = table1.merge(table1_, on=['机构'], how='left')
    table1_ = table1[['渠道', '变量', '渠道下平均iv', '渠道下最大psi', '渠道缺失率']].drop_duplicates(subset=['渠道', '变量'])
    table1_['flag'] = np.where((table1_.渠道下平均iv>0.02) & (table1_.渠道缺失率<0.95) & (table1_.渠道下最大psi<0.1), 1, 0)
    table1_ = table1_.groupby('渠道').apply(lambda x: pd.Series({
        '渠道':x['渠道'].iloc[0],
        '渠道下变量合格率': np.round(np.mean(x['flag']), 4),
        '渠道下变量合格个数':np.sum(x['flag'])
    })).reset_index(drop=True)
    table1 = table1.merge(table1_, on=['渠道'], how='left')
    
    return table1, res_iv_2, res_psi_0, res_psi_1, res_psi_2