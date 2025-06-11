def train_epoch_(org, tr_orgidx, val_orgidx, tr_idxs, val_idxs, X_tr, X_val, y_tr, y_val, w_tr, param):
    tr_idx = tr_orgidx.get(org)
    val_idx = val_orgidx.get(org)
    X_tr_, y_tr_, w_tr_ = X_tr.loc[list(tr_idxs-set(tr_idx)), ], y_tr.loc[list(tr_idxs-set(tr_idx)), ], w_tr.loc[list(tr_idxs-set(tr_idx)), ]
    X_val_, y_val_ = X_val.loc[list(val_idxs-set(val_idx)), ], y_val.loc[list(val_idxs-set(val_idx)), ]
    X_oos, y_oos = pd.concat([X_tr.loc[tr_idx, ], X_val.loc[val_idx, ]], axis=0) , pd.concat([y_tr.loc[tr_idx, ], y_val.loc[val_idx, ]], axis=0)
    callbacks = None
    if 'stopping_rounds' in param.keys():
        param.update({'num_iterations': 300})
        callbacks = [lgb.early_stopping(stopping_rounds=param.get('stopping_rounds'))]
    model = lgb.train(
                      param,
                      verbose_eval=0, 
                      train_set = lgb.Dataset(X_tr_, label=y_tr_, weight=w_tr_), 
                      valid_sets = [lgb.Dataset(X_tr_, label=y_tr_), lgb.Dataset(X_val_, label=y_val_)],
                      valid_names = ['train', 'val'],
                      callbacks = callbacks
                     )
    ks_tr, ks_val, ks_oos = _get_ks(model, X_tr_, y_tr_), _get_ks(model, X_val_, y_val_s), _get_ks(model, X_oos, y_oos)
    record = pd.DataFrame({'oos_org':org, 'train_ks':ks_tr, 'val_ks':ks_val, 'oos_ks':ks_oos}, index=['0'])
    return record

def train_epoch(X_tr, X_val, y_tr, y_val, w_tr, tr_orgidx, val_orgidx, param, fobj):
    if fobj is not None:
        param.update({'objective': fobj})
    tr_idxs, val_idxs = set(X_tr.index), set(X_val.index)
    results = pd.DataFrame()
    tasks = [(org, tr_orgidx, val_orgidx, tr_idxs, val_idxs, X_tr, X_val, y_tr, y_val, w_tr, param) for org in tr_orgidx.keys()]
    with Pool(5) as pool:
        records = pool.starmap(train_epoch_, tasks)
    for record in records:
        results = pd.concat([results, record], axis=0)
    return results
    

def gridsearch_params(params, data, max_interations,):
    feas = [v for v in data.columns if data[v].dtype!='O' and v!='new_target']
    tr_orgidx, val_orgidx, val_idx, tr_idx = {}, {}, [], []
    splitter = StratifiedShuffleSplit(n_splits=1, random_state=42, train_size=0.8)
    for org in data.new_org.unique():
        tmp_data = data[data.new_org==org].copy()
        for idx_tr, idx_val in splitter.split(tmp_data[feas], tmp_data['new_target']):
            tr_orgidx[org] = list(idx_tr)
            val_orgidx[org] = list(idx_val)
            val_idx += list(idx_val)
            tr_idx += list(idx_tr)
    data_tr, data_val = data.loc[tr_idx, ], data.loc[val_idx, ]
    X_tr, X_val, y_tr, y_val = data_tr[feas], data_val[feas], data_tr['new_target'], data_val['new_target']
    # replace with balanced func
    w_tr = pd.Series(np.ones(X_tr.shape[0]))
    
    good_params = pd.DataFrame()
    sampled_params = list(ParameterSampler(
        params, 
        n_iter=max_interations, 
        random_state=42
    ))
    begin_time_, begin_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), time.time()
    display(f"开始执行时间：{begin_time_}")
    for i, param in enumerate(tqdm.tqdm(sampled_params)):
        records = train_epoch(X_tr, X_val, y_tr, y_val, w_tr, tr_orgidx, val_orgidx, param, None)
        mean_tr_ks = np.mean(records['train_ks'])
        mean_val_ks = np.mean(records['val_ks'])
        mean_oos_ks = (np.sum(records['oos_ks'])-np.min(records['oos_ks'])-np.max(records['oos_ks']))*1.0 / (records.shape[0]-2)
        if np.allclose(records['train_ks'], records['val_ks'], rtol=3e-2):
            good_params = pd.concat([good_params, pd.DataFrame({'param': [param], 'mean_tr_ks':mean_tr_ks,
                                                                'mean_val_ks':mean_val_ks, 'mean_oos_ks':mean_oos_ks}, index=['0'])], axis=0)
            good_params_ = good_params[good_params.mean_val_ks==np.max(good_params.mean_val_ks)]
            display(f"当前最优参数下train平均ks是{np.round(good_params_['mean_tr_ks'].values, 3)}, val平均ks是{np.round(good_params_['mean_val_ks'].values, 3)}, oos平均ks是{np.round(good_params_['mean_oos_ks'].values, 3)}")
        current_time = time.time()
        display(f"平均每组参数训练耗时：{np.round((current_time-begin_time)*1.0 / ((i+1)*60), 2)}分")
    return good_params