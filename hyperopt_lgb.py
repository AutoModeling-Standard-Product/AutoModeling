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

class HyperOptLGB(object):
    '''
        init: data, params, fobj, max_iteration
        funcs: 
            split_data()
            train_epoch_()
            objective()
            tpesearch_params()
        return Trails
    '''
    def __init__(self, **kwargs):
        self.data = kwargs.get('data')
        self.params = kwargs.get('params')
        self.fobj = kwargs.get('fobj')
        self.max_iterations = kwargs.get('max_iterations')
        self.X_tr, self.X_val, self.y_tr, self.y_val, self.tr_orgidx, self.val_orgidx = self.split_data(self.data)
        # replace with weight-balance func, applied on entire dataset rather not trainset
        self.w_tr = pd.Series(np.ones(self.X_tr.shape[0]))
        self.trails = Trials()
    
    # 输入数据集，返回8:2切分的训练验证集 & 字典形式储存的各机构在训练集验证集的索引，确保输入的数据集有new_org, new_target列
    def split_data(self, data_):
        data = data_.copy()
        # 确定X，判定条件不是object类型且不是Y列
        feas = [v for v in data.columns if data[v].dtype!='O' and v!='new_target']
        # tr_orgidx存储训练集各个机构索引，tr_idx存储训练集全部索引, val为验证集同样做法
        tr_orgidx, val_orgidx, tr_idx, val_idx = {}, {}, [], []
        # 分层抽样
        splitter = StratifiedShuffleSplit(n_splits=1, random_state=42, train_size=0.8)
        for org in data.new_org.unique():
            tmp_data = data[data.new_org==org].copy()
            # 每个机构下分层抽样
            for idx_tr, idx_val in splitter.split(tmp_data[feas], tmp_data['new_target']):
                tr_orgidx[org] = list(idx_tr)
                val_orgidx[org] = list(idx_val)
                val_idx += list(idx_val)
                tr_idx += list(idx_tr)
        # 分出训练、验证集
        data_tr, data_val = data.loc[tr_idx, ], data.loc[val_idx, ]
        X_tr, X_val, y_tr, y_val = data_tr[feas], data_val[feas], data_tr['new_target'], data_val['new_target']
        return X_tr, X_val, y_tr, y_val, tr_orgidx, val_orgidx
    
    # 每组参数下分机构cv训练, 返回每个机构做oos下的train val oos ks, oos ks
    def train_epoch_(self, org, param):
        tr_idxs, val_idxs = set(self.X_tr.index), set(self.X_val.index)
        tr_idx, val_idx = self.tr_orgidx.get(org), self.val_orgidx.get(org)
        # 除去当前org选出训练验证集
        X_tr_, y_tr_ = self.X_tr.loc[list(tr_idxs-set(tr_idx)), ], self.y_tr.loc[list(tr_idxs-set(tr_idx)), ]
        # 获得训练集权重参数，weight后改为全部数据都有，根据训练集索引再取出需要的部分
        w_tr_ = self.w_tr.loc[list(tr_idxs-set(tr_idx)), ]
        X_val_, y_val_ = self.X_val.loc[list(val_idxs-set(val_idx)), ], self.y_val.loc[list(val_idxs-set(val_idx)), ]
        # 去除的机构当为oos
        X_oos, y_oos = pd.concat([self.X_tr.loc[tr_idx, ], self.X_val.loc[val_idx, ]], axis=0) , pd.concat([self.y_tr.loc[tr_idx, ], self.y_val.loc[val_idx, ]], axis=0)
        # 自动判断是否需要早停，如果用户参数中给出了早停则固定最大迭代次数为300，否则不设置早停
        callbacks = None
        if 'stopping_rounds' in param.keys():
            param.update({'num_iterations': 300})
            callbacks = [lgb.early_stopping(stopping_rounds=param.get('stopping_rounds'))]
        # 训练模型, 仅评估验证集
        model = lgb.train(
                          param,
                          verbose_eval=0, 
                          train_set = lgb.Dataset(X_tr_, label=y_tr_, weight=w_tr_), 
                          valid_sets = [lgb.Dataset(X_val_, label=y_val_)],
                          valid_names = ['train', 'val'],
                          callbacks = callbacks
                         )
        ks_tr, ks_val, ks_oos = _get_ks(model, X_tr_, y_tr_), _get_ks(model, X_val_, y_val_), _get_ks(model, X_oos, y_oos)
        record = pd.DataFrame({'oos_org':org, 'train_ks':ks_tr, 'val_ks':ks_val, 'oos_ks':ks_oos}, index=['0'])
        return record

    # 自定义目标函数，当参数符合要求时进一步更新超参数空间寻优
    def objective(self, param):
        begin_time = time.time()
        if self.fobj is not None:
            param.update({'objective': self.fobj})
        results = pd.DataFrame()
        # 开启9个进程池运行lgb
        tasks = [(org, param) for org in self.tr_orgidx.keys()]
        with Pool(9) as pool:
            records = pool.starmap(self.train_epoch_, tasks)
        for record in records:
            results = pd.concat([results, record], axis=0)
        
        mean_tr_ks = np.mean(results['train_ks'])
        mean_val_ks = np.mean(results['val_ks'])
        mean_oos_ks = (np.sum(results['oos_ks'])-np.min(results['oos_ks'])-np.max(results['oos_ks']))*1.0 / (results.shape[0]-2)
        # 判断参数符合要求条件为每个机构做oos时的训练集和验证集ks差距在相对3%以下，否则不更新loss
        if np.allclose(results['train_ks'], results['val_ks'], rtol=3e-2):
            loss = -(0.5*mean_val_ks + 0.5*mean_oos_ks)
            status = STATUS_OK
        else:
            loss = np.Inf
            status = STATUS_FAIL
        end_time = time.time()
        display(f"当前组参数训练耗时：{np.round((end_time-begin_time)*1.0/60, 2)}分")
        return {'loss': loss, 'param':param, 'status':status}
    
    # 该类的执行函数，返回trails
    def tpesearch_params(self):
        begin_time_ = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        display(f"开始执行时间：{begin_time_}")
        _ = fmin(fn=self.objective, space=self.params, algo=tpe.suggest, max_evals=self.max_iterations, trials=self.trails)
        return pd.DataFrame(self.trails)