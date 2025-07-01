def trans_score_emerginguser_vectorized(x, m, b):
    
    x = np.where((x <= 0) | (x >= 1), 0, x)
    
    tmp = np.log(x / (1-x))
    min_ = np.min(tmp)
    max_ = np.max(tmp)
    
#     m, b = np.polyfit([min_, max_], [900, 300], 1)
    m = -114.29489810948542
    b = 388.6946625498659
    
    ret = np.round(m * tmp + b, 0).astype(int)
    ret = np.where(ret < 300, 300, ret)
    ret = np.where(ret > 900, 900, ret)
    
    return ret


def calculate_scorebin10(x1=None, Y=None, data=None, data_Y=None):
    import numpy as np
    assert x1 in data.columns and Y in data.columns, "列名不匹配"
    assert Y in data_Y.columns, "列名不匹配"
    data = data[[x1, Y]]
    data_Y = data_Y[[Y]]
    data[x1] = data[x1].astype(float)
    data[Y] = data[Y].astype(int)
    data = data.sort_values(by=x1)
    bad_cnt_total = sum(data[Y])
    good_cnt_total = len(data[Y])-bad_cnt_total
    bad_rate_total = np.mean(data[Y])
    # 计算标签样本Y下好坏样本综合
    bad_cnt_total_dataY = sum(data_Y[Y])
    good_cnt_total_dataY = len(data_Y) - bad_cnt_total_dataY
    cnt_total_dataY = len(data_Y)
    
    combiner = toad.transform.Combiner()
    combiner.fit(data[[x1, Y]], y=Y, method='chi', n_bins=10)
    data['bin'] = combiner.transform(data[[x1]])[[x1]]
    data['bin'] = data['bin'].astype(str)
    
    res = data.groupby('bin', as_index=False).apply(
        lambda tmp: pd.DataFrame({
            'name': x1,
            'bin': tmp['bin'],
            'cnt_total': len(tmp),
            'good_cnt': len(tmp)-sum(tmp[Y]),
            'bad_cnt': sum(tmp[Y]),
            'hit_rate': round(len(tmp)*1.0/cnt_total_dataY, 4),
            'bad_rate': round(np.mean(tmp[Y]), 4),
            'lift': round(np.mean(tmp[Y])*1.0/ bad_rate_total, 4),
            'recall': round(sum(tmp[Y])*1.0/bad_cnt_total_dataY ,4),
            'accident_rate': round((len(tmp)-sum(tmp[Y]))*1.0/good_cnt_total_dataY, 4),
            'ks_bin': round(max(abs(tmp[Y].cumsum()*1.0/bad_cnt_total-
                            abs(1-tmp[Y]).cumsum()*1.0/good_cnt_total)), 4),
            'iv_bin': round(((len(tmp)-sum(tmp[Y]))/good_cnt_total - sum(tmp[Y])/bad_cnt_total)*
                            np.log(((len(tmp)-sum(tmp[Y]))/good_cnt_total) / (1e-10+
                                                                            sum(tmp[Y])/bad_cnt_total)), 4)
        })
    )
    
    res = pd.DataFrame(res).drop_duplicates()
    
    res['bad_cnt_agg_bin'] =  res['bad_cnt'].cumsum()
    res['good_cnt_agg_bin'] =  res['good_cnt'].cumsum()
    res['cnt_total_agg_bin'] = res['cnt_total'].cumsum()
    
    res = res.rename(columns={'bin':'分箱', 'cnt_total':'样本量', 'hit_rate':'命中率',
                       'good_cnt':'好样本', 'bad_cnt':'怀样本', 'bad_rate':'怀样本占比',
                       'lift': '提升度', 'recall':'召回率', 'accident_rate': '误伤率',
                       'ks_bin': '分箱ks', 'iv_bin': '分箱IV'})
    
    res['分箱IV'].replace({np.inf: 0}, inplace=True)
    res['总IV'] = sum(res['分箱IV'])
    res['累积命中率'] = res['命中率'].cumsum()
    res['累积提升度'] = round((res['bad_cnt_agg_bin']/res['cnt_total_agg_bin'])*1.0/bad_rate_total, 4)
    res['累积召回率'] = round(res['bad_cnt_agg_bin']*1.0/bad_cnt_total_dataY, 4)
    res['累积误伤率'] = round(res['good_cnt_agg_bin']*1.0/good_cnt_total_dataY, 4)
    
    res.drop(columns=['bad_cnt_agg_bin', 'good_cnt_agg_bin','cnt_total_agg_bin'], inplace=True)
    
    return res

def get_result_excels(data, data_Y, file_name, flag=True):
    
    data_Y['apply_date'] = data_Y['apply_date'].astype(str).str[0:6]
    
    # 查看样本Y情况
    statis_Y = data_Y.groupby('apply_date').apply(
        lambda s: pd.DataFrame({
            'ym': s['apply_date'],
            'total_cnt': len(s),
            'bad_cnt': sum(s['credit_target']),
            'bad_percent': round(1.0*sum(s['credit_target'])/len(s), 4)
        })
    )
    statis_Y = pd.DataFrame(statis_Y).sort_values(by='ym')
    statis_Y = statis_Y.drop_duplicates()

    null_ = data.isnull().sum()
    len_Y = len(data_Y)
    len_data= len(data)
    hit_rates = []
    for col in data.columns:
        if col not in ['credit_target']:
            na_cnt = data[col].isna().sum()
            hit_rate = round((len_data-na_cnt)*1.0/len_Y, 4)
            hit_rates.append({'name':col, 'hit_rate': hit_rate, 'na_cnt': na_cnt})
    hit_rates = pd.DataFrame(hit_rates).sort_values(by='hit_rate')

    # 首先运行auc,判断x,y正负相关为后续计算lift提供信息
    auc_res = []
    for col in data.columns:
        if col not in ['credit_target', 'apply_date']:
            try:
                auc_tmp = calculate_auc(col, 'credit_target', data)
                auc_res.append(auc_tmp)
            except:
                print("{}计算auc出错，跳过".format(col))
                continue
    
    auc_res_df = pd.DataFrame(auc_res)
    
    # 再运行ks后续与lift拼接
    ks_res_df = pd.DataFrame()
    for col in data.columns:
        if col not in ['credit_target']:
            try:
                ks_tmp = calculate_ks(col, 'credit_target', data)
                ks_res_df = pd.concat([ks_res_df, ks_tmp],axis=0)
            except:
                print("{}计算ks出错，跳过".format(col))
                continue
    
    lift_res_df = pd.DataFrame()
    for col in data.columns:
        if col not in ['credit_target']:
            try:
                flag = True
                flag_val = auc_res_df[auc_res_df['name']==col]['flag'].values
                if flag_val==1:
                    flag = False
                else:
                    flag = True
                res = calculate_lift(col, 'credit_target', flag, data)
                tmp = pd.DataFrame(
                    {
                        'name': res[0]['name'].values,
                        '1%提升度': res[0]['1%lift'].values, '2%提升度': res[1]['2%lift'].values, '3%提升度': res[2]['3%lift'].values,
                        '5%提升度': res[3]['5%lift'].values,'10%提升度': res[4]['10%lift'].values, '15%提升度': res[5]['15%lift'].values}
                )
                lift_res_df = pd.concat([lift_res_df, tmp], axis=0)
            except:
                print("{}计算lift出错，跳过".format(col))
        
    # 最后运行scorebin10得到每个子分10等频分箱的信息
    scorebin10_res_df = pd.DataFrame()
    for col in data.columns:
        if col not in ['credit_target']:
            try:
                scorebin10_tmp = calculate_scorebin10(col, 'credit_target', data, data_Y)
                scorebin10_res_df = pd.concat([scorebin10_res_df, scorebin10_tmp],axis=0)
            except:
                print("{}计算scorebin出错，跳过".format(col))
                continue
    
    tmp = pd.merge(hit_rates, ks_res_df,on=['name'])
    tmp = tmp[['name', 'hit_rate', 'ks']].drop_duplicates()
    tmp = pd.merge(tmp, auc_res_df, on=['name'])
    tmp = tmp[['name', 'hit_rate', 'ks', 'total_auc']].drop_duplicates()
    table2 = pd.merge(tmp, lift_res_df, on=['name']).drop_duplicates()
    scorebin10_res_df_iv = scorebin10_res_df[['name','总IV']]
    table2 = pd.merge(table2, scorebin10_res_df_iv, on='name', how='left').drop_duplicates()
    # 将IV列放在AUC后
    order = [col for col in table2.columns if col!='总IV']
    order.insert(4, '总IV')
    table2 = table2.reindex(columns=order)
    table2 = table2.sort_values(by=['ks'], ascending=False)
    
    tmp_ks_res_df = ks_res_df[['name', 'ks']]
    table3 = pd.merge(scorebin10_res_df, tmp_ks_res_df, on=['name'])
    table3 = table3.drop_duplicates()
    
    flag = True
    if flag:
        ## 表1
        writer = pd.ExcelWriter(file_name+'.xlsx', engine='xlsxwriter')
        statis_Y.to_excel(writer, sheet_name='坏样概况', index=False)

        # 表2
        table2.to_excel(writer, sheet_name='子分效果总览', index=False)

        # 表3
        table3.to_excel(writer, sheet_name='评分等频10箱', index=False)

        writer.save()
        writer.close()
    
    return statis_Y, table2, table3