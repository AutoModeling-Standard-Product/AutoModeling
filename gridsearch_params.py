def top_5_lift(pred_, data):
    y = data.get_label()
    pred = 1 / (1 + np.exp(-pred_))
    lift5 = _get_lift(y, pred, 0.05)
    return '5%lift', lift5, True
def top_10_lift(pred_, data):
    y = data.get_label()
    pred = 1 / (1 + np.exp(-pred_))
    lift10 = _get_lift(y, pred, 0.1)
    return '10%lift', lift10, True 
def _get_lift(y, pred, k):
        n_top = int(len(y) * k)
        top_indices = pd.Series(pred).sort_values(ascending=False).head(n_top).index
        return y[top_indices].mean() / y.mean()
def _get_ks(model, X, y):
        pred = model.predict(X)
        ks = toad.metrics.KS(pred, y)
        return ks

def train_epoch(X_tr, X_val, y_tr, y_val, w_tr, tr_orgidx, val_orgidx, param, fobj):
    if fobj is not None:
        param.update({'objective': fobj})
    tr_idxs = set(X_tr.index)
    val_idxs = set(X_val.index)
    records = pd.DataFrame()
    for org in tr_orgidx.keys():
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
        ks_tr, ks_val, ks_oos = _get_ks(model, X_tr_, y_tr_), _get_ks(model, X_val, y_val), _get_ks(model, X_oos, y_oos)
        records = pd.concat([records, pd.DataFrame({'oos_org':org, 'train_ks':ks_tr, 'val_ks':ks_val, 'oos_ks':ks_oos}, index=['0'])], axis=0)
    mean_tr_ks = np.mean(records['train_ks'])
    mean_val_ks = np.mean(records['val_ks'])
    mean_oos_ks = (np.sum(records['oos_ks'])-np.min(records['oos_ks'])-np.max(records['oos_ks']))*1.0 / (records.shape[0]-2)
    return mean_tr_ks, mean_val_ks, mean_oos_ks

def gridsearch_params(params, data, max_interations, max_gap, min_ks):
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
        mean_tr_ks, mean_val_ks, mean_oos_ks = train_epoch(X_tr, X_val, y_tr, y_val, w_tr, tr_orgidx, val_orgidx, param, None)
        if abs(mean_tr_ks-mean_val_ks)<=max_gap and mean_val_ks>=min_ks:
            good_params = pd.concat([good_params, pd.DataFrame({'param': param, 'mean_tr_ks':mean_tr_ks,
                                                                'mean_val_ks':mean_val_ks, 'mean_oos_ks':mean_oos_ks}, index=['0'])], axis=0)
            good_params_ = good_params[good_params.mean_val_ks==np.max(good_params.mean_val_ks)]
            display(f"当前最优参数下train平均ks是{np.round(good_params_['mean_tr_ks'], 3)}, val平均ks是{np.round(good_params_['mean_val_ks'], 3)}, oos平均ks是{np.round(good_params_['mean_oos_ks'], 3)}")
        current_time = time.time()
        display(f"平均每组参数训练耗时：{np.round((current_time-begin_time)*1.0 / ((i+1)*60), 2)}分")
    return good_params