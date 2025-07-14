from utils.requirements import *

def org_analysis(data: pd.DataFrame)-> pd.DataFrame:
    assert all(v in data.columns for v in ['new_org', 'new_date', 'new_date_ym', 'new_target']), "输入的数据列不符合命名要求"
    try:
        datasetStatis = data[['new_org', 'new_date_ym', 'new_target']].groupby(['new_org', 'new_date_ym']).apply(lambda x:pd.Series({
            '机构': x['new_org'].iloc[0],
            '年月': x['new_date_ym'].iloc[0],
            '单月坏样本数': int(np.sum(x['new_target'])),
            '单月总样本数': int(len(x['new_target'])),
            '单月坏样率': np.round(np.mean(x['new_target']), 3)
        }))
        datasetStatis = pd.DataFrame(datasetStatis)
        datasetStatis['总坏样本数'] = datasetStatis.groupby('机构')['单月坏样本数'].transform('sum').astype(int)
        datasetStatis['总样本数'] = datasetStatis.groupby('机构')['单月总样本数'].transform('sum')
        datasetStatis['总坏样率'] = np.round(1.0 * datasetStatis['总坏样本数'] / datasetStatis['总样本数'], 3)
        datasetStatis = datasetStatis.reset_index(drop=True)
        datasetStatis = datasetStatis[['机构', '年月', '单月坏样本数', '单月总样本数', '单月坏样率',
                                      '总坏样本数', '总样本数', '总坏样率']].sort_values(by=['机构', '年月'])
    except Exception as e:
        print("机构样本分析出现错误"+e)
    return datasetStatis

def missing_check(**kwargs) -> pd.DataFrame:
    data = kwargs.get('data').copy()
    assert all(v in data.columns for v in ['new_org', 'new_date', 'new_date_ym', 'new_target']), "输入的数据列不符合命名要求"
    channels = kwargs.get('channel')
    data = {
        channel: data[data.new_org.isin(orgs)] for channel, orgs in channels.items()
    }
    miss_res = pd.DataFrame()
    miss_res1 = pd.DataFrame()
    for channel in tqdm.tqdm([v for v in channels.keys()]):
        sub_data = data.get(channel).copy()
        for col in set(sub_data.columns)-set(['new_date', 'new_date_ym', 'new_target', 'new_org']):
            if channel == "整体":
                try:
                    tmp_res = sub_data[[col, 'new_org', 'new_target']].groupby('new_org').apply(lambda x: pd.DataFrame({
                        '机构':x['new_org'].iloc[0],
                        '变量': col,
                        '-1111缺失率': np.round((x[col]==-1111).mean(), 3),
                        '-999缺失率': np.round((x[col]==-999).mean(), 3),
                        '-1缺失率': np.round((x[col]==-1).mean(), 3),
                        'nan缺失率': np.round(x[col].isna().mean(), 3),
                         ## 总缺失率为0 bug
                        '总缺失率': np.round(((x[col]==-1)|(x[col]==-999)|(x[col]==-1111)|(x[col]==np.nan)).mean(), 3)
                    }, index=['0']))
                    miss_res = pd.concat([miss_res, tmp_res], axis=0)
                except Exception as e:
                    print(f"机构{org}下{col}计算缺失率失败, {e}")
            try:
                tmp_res_ = pd.DataFrame({'渠道': channel, 
                                        '变量': col, 
                                        '-1111缺失率': np.round((sub_data[col]==-1111).mean(), 3),
                                        '-999缺失率': np.round((sub_data[col]==-999).mean(), 3),
                                        '-1缺失率': np.round((sub_data[col]==-1).mean(), 3),
                                        'nan缺失率': np.round(sub_data[col].isna().mean(), 3),
                                        '总缺失率': np.round(((sub_data[col]==-1)|(sub_data[col]==-999)|(sub_data[col]==-1111)|(sub_data[col]==np.nan)).mean(), 3)}, index=['0'])
                miss_res1 = pd.concat([miss_res1, tmp_res_], axis=0)
            except Exception as e:
                print(f"渠道{channel}下计算{col}缺失率失败, {e}")
    miss_res = miss_res.reset_index(drop=True).drop_duplicates(subset=['机构', '变量']).sort_values(by=['机构', '总缺失率'], ascending=False)
    miss_res1 = miss_res1.reset_index(drop=True).sort_values(by=['渠道', '总缺失率'], ascending=False)
    return miss_res, miss_res1


def get_dataset(**kwargs) -> pd.DataFrame:
    data_pth = kwargs.get("data_pth")
    date_colName = kwargs.get("date_colName")
    y_colName = kwargs.get("y_colName")
    org_colName = kwargs.get("org_colName")
    data_encode = kwargs.get("data_encode")
    key_colNames = kwargs.get("key_colNames")
#     use_cols = kwargs.get("use_cols")
    assert all(v is not None for v in (data_pth, date_colName, y_colName, data_encode)), '仅允许输入中org_colName为None'
    assert type(key_colNames) == list, '主键key_colNames必须为list类型'
    
    try:
        data = pd.read_csv(data_pth, parse_dates=[date_colName], encoding=data_encode)
    except Exception as e1:
        try:
            data = pd.read_xlsx(data_pth, parse_dates=[date_colName], encoding=data_encode)
        except Exception as e2:
            try:
                print("pickle文件无法应用use_cols, 读取全部数据")
                data = pd.read_pickle(data_pth)
            except Exception as e3:
                print(e1+"   "+e2+"   "+e3)
                return
    try:
        print(f'原始数据有{data.shape[0]}条, 根据{key_colNames}去重且只保留标签列[0,1]的数据')
        data = data[data[y_colName].isin([0, 1])]
        data = data.drop_duplicates(subset=key_colNames)
        print(f'去重后数据有{data.shape[0]}条')
        
        if 'Unnamed: 0' in data.columns:
            data.drop(columns=['Unnamed: 0'], inplace=True)
        
        for fea in data.columns:
            if data[fea].nunique()<=1:
                print(f"{fea}全为{data[fea].iloc[0]}, 去除该列")
                data.drop(columns=fea, inplace=True)
        
        if org_colName is None:
            data[org_colName] = 'unique'
            print(f'输入对的org_colName为空, 将该列设置为唯一值[unique]')
        
        print(f"{y_colName}, {org_colName}被重命名为new_target, new_org; {date_colName}被格式化为new_date, new_date_ym两列")
        data.rename(columns={date_colName:'new_date', y_colName:'new_target', org_colName:'new_org'}, inplace=True)
        data['new_date'] = data['new_date'].astype(str).apply(lambda x: str(x).replace('-', '')[:8])
        data['new_date_ym'] = data['new_date'].apply(lambda x: str(x)[:6])
        data = data.reset_index(drop=True)
    except Exception as e:
        print(e+'数据获取失败')
        return None
    
    display(f"原始数据大小为{data.shape}")
    display(data.head(1))
    return data


def calculate_psi(base, current):
    sorted_base = sorted(base)
    n = len(sorted_base)
    edges = []
    for p in np.arange(0, 110, 10):
        index = (n-1) * p / 100.0
        if index.is_integer():
            edges.append(sorted_base[int(index)])
        else:
            lower_idx = int(index)
            upper_idx = lower_idx + 1
            fraction = index - lower_idx
            edges.append(sorted_base[lower_idx]+fraction*(sorted_base[upper_idx]-sorted_base[lower_idx]))
    
    # 扩展 current，将在edges外的数据归于距离最近的一箱
    current_clipped = np.clip(current, edges[0], edges[-1])
    base_cnt, _ = np.histogram(base, bins=edges)
    current_cnt, _ = np.histogram(current_clipped, bins=edges)
    base_percentage = 1.0 * base_cnt / base_cnt.sum()
    current_percentage = 1.0 * current_cnt / current_cnt.sum()
    
    psi = 0
    for i in range(len(base_percentage)):
        if base_percentage[i]==0:
            base_percentage[i] = 1e-6
        if current_percentage[i]==0:
            current_percentage[i] = 1e-6
        psi += (current_percentage[i]-base_percentage[i]) * np.log(current_percentage[i] / base_percentage[i])
    
    return np.round(psi, 4)
    
def detect_psi(**kwargs)-> pd.DataFrame:
    data = kwargs.get("data").copy()
    assert all(v in data.columns for v in ['new_org', 'new_date', 'new_date_ym', 'new_target']), "输入的数据列不符合命名要求"
    data.replace({-1111:-999, -1:-999, np.nan:-999}, inplace=True)
    data = data[data.new_target.isin([0, 1])]
    data['new_date'] = data['new_date'].astype(int)
    channels = kwargs.get('channel')
    data = {
        channel: data[data.new_org.isin(orgs)] for channel, orgs in channels.items()
    }
    
    psi_res = []
    psi_res1 = []
    for channel in tqdm.tqdm([v for v in data.keys()]):
        print(f"当前执行渠道{channel}")
        sub_data = data.get(channel)
        
        ## 如果只有渠道为整体时执行分机构查看,默认输入的渠道一定有整体
        if channel == '整体':
            print("逐机构计算psi")
            for org in sub_data.new_org.unique():
                try:
                    data_tmp = sub_data[sub_data.new_org==org].copy()

                    if len(data_tmp.new_date.unique()) < 4:
                        # 根据日期大小赋予递增唯一值
                        ranked_date = data_tmp['new_date'].rank(method='first')
                        data_tmp['new_date_bin'] = pd.qcut(ranked_date, q=4, labels=[0, 1, 2, 3])
                    else:
                        try:
                            data_tmp['new_date_bin'] = pd.qcut(data_tmp['new_date'], q=4, labels=[0, 1, 2, 3])
                        except:
                            data_tmp['new_date_bin'] = pd.qcut(data_tmp['new_date'], q=5, labels=[0, 1, 2, 3], duplicates='drop')
                            assert len(data_tmp['new_date_bin'].unique()) == 4, f"{org}下4分箱失败"

                    data_tmp_ = data_tmp.groupby('new_date_bin')['new_date'].agg(['count'])
                    for col in list(set(data_tmp.columns)-set(['new_org', 'new_date', 'new_date_ym', 'new_target', 'new_date_bin'])):
                        if data_tmp[col].dtype != 'O':
                            for idx in np.arange(0, 4):
                                base = data_tmp[data_tmp.new_date_bin==idx][col]
                                current = data_tmp[data_tmp.new_date_bin==min(idx+1, 3)][col]
                                psi_tmp = calculate_psi(base, current)
                                psi_res.append({'机构':org, 
                                                '变量':col, 
                                                '区间': "Q"+str(idx+1),
                                                '样本数':int(data_tmp_[data_tmp_.index==idx]['count'].values), 
                                                '区间psi': psi_tmp})
                except Exception as e:
                    print(f"机构{org}下{col}计算psi失败, {e}")
        
        if len(sub_data.new_date.unique()) < 4:
            # 根据日期大小赋予递增唯一值
            ranked_date = sub_data['new_date'].rank(method='first')
            sub_data['new_date_bin'] = pd.qcut(ranked_date, q=4, labels=[0, 1, 2, 3])
        else:
            sub_data['new_date_bin'] = pd.qcut(sub_data['new_date'], q=4, labels=[0, 1, 2, 3])
        
        sub_data_ = sub_data.groupby('new_date_bin')['new_date'].agg(['count'])
        for col in list(set(sub_data.columns)-set(['new_org', 'new_date', 'new_date_ym', 'new_target', 'new_date_bin'])):
            try:
                if sub_data[col].dtype != 'O':
                    for idx in np.arange(0, 4):
                        base = sub_data[sub_data.new_date_bin==idx][col]
                        current = sub_data[sub_data.new_date_bin==min(idx+1, 3)][col]
                        psi_tmp = calculate_psi(base, current)
                        psi_res1.append({'渠道':channel, 
                                        '变量':col, 
                                        '区间': "Q"+str(idx+1),
                                        '样本数':int(sub_data_[sub_data_.index==idx]['count'].values), 
                                        '区间psi': psi_tmp})
            except Exception as e:
                print(f'渠道{channel}下{col}计算失败, {e}')
    
    psi_res = pd.DataFrame(psi_res)
    psi_res1 = pd.DataFrame(psi_res1)
    psi_res['最大psi'] = psi_res.groupby(['机构', '变量'])['区间psi'].transform('max')
    psi_res1['最大psi'] = psi_res1.groupby(['渠道', '变量'])['区间psi'].transform('max')
    psi_res = psi_res.drop_duplicates(subset=['机构','变量', '区间']).sort_values(by=['机构', '变量', '区间','最大psi'], ascending=False)  
    psi_res1 = psi_res1.sort_values(by=['渠道', '变量', '区间','最大psi'], ascending=False)    
    return psi_res, psi_res1

def calculate_iv(x: pd.Series, y: pd.Series, method: str, bins: int) -> Tuple[float, pd.Series, pd.Series]:
    data = pd.concat([x, y], axis=1)
    data.columns = ['x', 'y']
    assert method in ['dt', 'quantile'], "分箱方法必须是dt或quantile"
    assert type(bins) is int and bins>=2, '分箱数最小为2'
    
    c = toad.transform.Combiner()
    if method == "dt":
        data['x_bin'] = c.fit_transform(X=data['x'], y=data['y'], method='dt', **dict(n_bins=bins, min_samples=0.5/bins, empty_separate=True))
    elif method == "quantile":
        data['x_bin'] = c.fit_transform(X=data['x'], y=data['y'], method='quantile', **dict(n_bins=bins, empty_separate=True))
    
    iv = toad.quality(data[['x_bin', 'y']], 'y', iv_only=True).loc['x_bin', 'iv']
    
    return np.round(iv, 4), c.export(), data['x_bin']

def trend_detect(x: pd.Series, y: pd.Series, egde_dict: dict) -> Tuple[float, pd.Series]:
    data = pd.concat([x, y], axis=1)
    data.columns = ['x', 'y']
    
    c = toad.transform.Combiner()
    c.load(egde_dict)
    data = c.transform(data[['x', 'y']], labels=True)
    
    woe_transformer = toad.transform.WOETransformer()
    woe_transformer.fit(data[['x', 'y']], y='y')
    woe = pd.Series(woe_transformer.transform(data['x']))
    iv = toad.quality(data[['x', 'y']], 'y', iv_only=True).loc['x', 'iv']
    
    bin_plot(data, x='x', target='y')
    fig = plt.gcf()
    plt.close()
    
    return iv, woe, fig

def detect_iv(**kwargs) -> pd.DataFrame:
    data = kwargs.get("data").copy()
    assert all(v in data.columns for v in ['new_org', 'new_date', 'new_date_ym', 'new_target']), "输入的数据列不符合命名要求"
    data.replace({-1111: np.nan, -999: np.nan}, inplace=True)
    data = data[data.new_target.isin([0, 1])]
    channels = kwargs.get('channel')
    data = {channel: data[data.new_org.isin(orgs)] for channel, orgs in channels.items()}
    method = kwargs.get('method')
    bins = kwargs.get('bins')
    
    res = []
    res_1 = []
    for channel in tqdm.tqdm([v for v in channels.keys()]):
        sub_data = data.get(channel).copy()
        if channel == '整体':
            print(f"单机构计算iv")
            for org in sub_data.new_org.unique():
                tmp_data = sub_data[sub_data.new_org==org].copy()
                for col in list(set(tmp_data.columns)-set(['new_org', 'new_date', 'new_date_ym', 'new_target'])):
                    if tmp_data[col].dtype != 'O':
                        try:
                            iv, _, _ = calculate_iv(tmp_data[col], tmp_data['new_target'], method, bins)
                            res.append({'机构':org, '变量': col, 'iv': iv})
                        except Exception as e:
                            res.append({'机构':org, '变量': col, 'iv': 0})
                            print(f"机构{org}下{col}计算iv失败, {e}, 赋值相应iv为0")
        
        for col in list(set(sub_data.columns)-set(['new_org', 'new_date', 'new_date_ym', 'new_target'])):
             if sub_data[col].dtype != 'O':
                try:
                    iv, _, _ = calculate_iv(sub_data[col], sub_data['new_target'], method, bins)
                    res_1.append({'渠道':channel, '变量': col, 'iv': iv})
                except Exception as e:
                    print(f"渠道{channel}下{col}计算iv失败, {e}")
    
    res = pd.DataFrame(res).sort_values(by=['机构', 'iv'], ascending=False)
    res_1 = pd.DataFrame(res_1).sort_values(by=['渠道', 'iv'], ascending=False)
    return res, res_1

def detect_correlation(**kwargs) -> pd.DataFrame:
    data = kwargs.get('data').copy()
    method = kwargs.get('method')
    max_corr = kwargs.get('max_corr')
    data.replace({-1111: np.nan, -999:np.nan, -1:np.nan}, inplace=True)
    features = [v for v in data.columns if data[v].dtype!='O']
    corrs = data[features].corr(method=method)
    features_y = []
    if 'new_target' in corrs.columns:
        corr_y = pd.DataFrame(corrs.loc['new_target', :,]).reset_index()
        corr_y.columns = ['feature', 'correlation']
        features_y = list(set(corr_y[(abs(corr_y.correlation)>0.5)]['feature'])-set(['new_target']))
    if len(features_y)>0:
        print(f'{features_y}与new_target列相似性超过0.5')
    corrs.pop('new_target')
    corrs = corrs[~corrs.index.isin(['new_target'])]
    corr = pd.DataFrame(
        np.where((abs(corrs)>max_corr)&(corrs!=1), 1, 0),
        columns=corrs.columns
    )
    corr.index = corr.columns
    indices = corr.where(np.triu(corr, k=1)==1).stack().index.tolist()
    return indices, corrs

def get_fixed_lgb(max_depth=5, n_estimators=100)->lgb.LGBMClassifier:
    params_dict = {
        'objective':'binary',
        'split':'gain',
        'learning_rate':0.05,
        'max_depth':max_depth,
        'min_child_samples':20,
        'min_child_weight':20,
        'n_estimators':n_estimators,
        'num_leaves':2^max_depth - 1}
    clf = lgb.LGBMClassifier(**params_dict)
    return clf