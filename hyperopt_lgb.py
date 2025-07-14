from main.utils.requirements import *
from main.utils.data_augmentation import *
from main.utils.Inference import *

class HyperOptLGB(object):
    '''
        init: data, params, fobj, weights, max_iteration
        
        excuting func: tpesearch_params()
        
        return: train log, trails
    '''
    def __init__(self, **kwargs):
        self.data = kwargs.get('data')
        self.params = kwargs.get('params')
        self.fobj = kwargs.get('fobj')
        self.max_iterations = kwargs.get('max_iterations')
        self.record_train_process = kwargs.get("record_train_process")
        self.auc_threshold = kwargs.get('auc_threshold')
        self.randn = kwargs.get('randn')
        self.X_tr, self.X_val, self.y_tr, self.y_val, self.tr_orgidx, self.val_orgidx = self.split_data(self.data)
        self.trails = Trials()
    
    # 输入数据集，返回8:2切分的训练验证集 & 字典形式储存的各机构在训练集验证集的索引，确保输入的数据集有new_org, new_target列
    def split_data(self, data):
        
        # 确定X，判定条件不是object类型且不是Y列
        feas = [v for v in data.columns if data[v].dtype!='O' and v!='new_target']
        
        # tr_orgidx存储训练集各个机构索引，tr_idx存储训练集全部索引, val为验证集同样做法
        tr_orgidx, val_orgidx, tr_idx, val_idx = {}, {}, [], []
        
        # 分层抽样
        splitter = StratifiedShuffleSplit(n_splits=1, random_state=self.randn, train_size=0.8)
        
        for org in data.new_org.unique():
            tmp_data = data[data.new_org==org].copy()
            org_index = tmp_data.index
            
            # 每个机构下分层抽样, 注意不要使用相对索引否则会造成取值错误, 这里使用了绝对索引
            for idx_tr, idx_val in splitter.split(tmp_data[feas], tmp_data['new_target']):
                tr_orgidx[org] = list(org_index[idx_tr])
                val_orgidx[org] = list(org_index[idx_val])
                val_idx += list(org_index[idx_val])
                tr_idx += list(org_index[idx_tr])
        
        # 分出训练、验证集
        data_tr, data_val = data.loc[tr_idx, ], data.loc[val_idx, ]
        X_tr, X_val, y_tr, y_val = data_tr[feas], data_val[feas], data_tr['new_target'], data_val['new_target']
        return X_tr, X_val, y_tr, y_val, tr_orgidx, val_orgidx
    
    def _get_lift(self, y, pred, k):
        
        n_top = int(len(y) * k)
        top_indices = pd.Series(pred, index=y.index).sort_values(ascending=False).head(n_top).index
        
        return y[top_indices].mean() / y.mean()
            
    ## 自定义feval评价函数，闭包方式，输入加权&不加权ks auc等, lightgbm==2.3.0时每次迭代都会强制评估无法跳过, 每次评估约为0.2-0.8秒均值为0.3秒
    def weighted_metric(self, weights):
        
        def _weighted_metric(pred_, data):
            y = data.get_label()
            pred = 1 / (1 + np.exp(-pred_))
            
            ## 比较输入的数据长度和w_val_, w_tr_, w_oos_判断使用哪个权重
            if len(y) == len(weights.get('train')):
                w = weights.get('train')
            elif len(y) == len(weights.get('val')):
                w = weights.get('val')
            else:
                w = weights.get('oos')
                
            auc_w= roc_auc_score(y, pred, sample_weight=w)
            fpr, tpr, _ = roc_curve(y, pred, sample_weight=w)
            
            ks = toad.metrics.KS(pred, y)
            ks_w = max(tpr - fpr)
            
            lift10, lift5 = self._get_lift(y, pred, 0.1), self._get_lift(y, pred, 0.05)
            
            return [
                    ('auc_w',auc_w,True),
                    ('ks_w',ks_w,True), ('ks',ks,True),
                    ('5%lift',lift5,True), ('10%lift',lift10,True)
                ]
        
        return _weighted_metric
    
    ## 根据model和输入数据返回带权重&不带权重计算下的auc ks，替代weighted_metric用，只计算最后一次
    def single_weighted_metric(self, model, X, y, w):
        pred = model.predict(X)
        pred = pd.Series(pred, index=X.index)
        auc_w, auc = roc_auc_score(y, pred, sample_weight=w), roc_auc_score(y, pred)
        fpr, tpr, _ = roc_curve(y, pred, sample_weight=w)
        
        ks = toad.metrics.KS(pred, y)
        ks_w = max(tpr - fpr)
        
        lift10, lift5 = self._get_lift(y, pred, 0.1), self._get_lift(y, pred, 0.05)
        
        return auc_w, auc, ks_w, ks, lift5, lift10
    
    ## 提取训练日志中的最后一次迭代结果
    def extract_evalresult(self, records):
        
        simpler_records = []
        
        for org in records.org.unique():
            tmp_record = records[records.org==org].copy()
            
            simpler_records.append({
                'org': org, 'tr_auc':tmp_record.train.auc[-1], 'tr_auc_w':tmp_record.train.auc_w[-1],
                'tr_ks':tmp_record.train.ks[-1], 'tr_ks_w': tmp_record.train.ks_w[-1],
                'tr_5lift':tmp_record.train['5%lift'][-1], 'tr_10lift':tmp_record.train['10%lift'][-1],
                'val_auc':tmp_record.val.auc[-1], 'val_auc_w':tmp_record.val.auc_w[-1],
                'val_ks':tmp_record.val.ks[-1], 'val_ks_w':tmp_record.val.ks_w[-1],
                'val_5lift':tmp_record.val['5%lift'][-1], 'val_10lift':tmp_record.val['10%lift'][-1],
                'oos_auc': tmp_record.oos.auc[-1], 'oos_ks':tmp_record.oos.ks[-1],
                'oos_5lift':tmp_record.oos['5%lift'][-1], 'oos_10lift':tmp_record.oos['10%lift'][-1]
                })
        
        simpler_records = pd.DataFrame(simpler_records)
        
        return simpler_records
    
    # 每组参数下分机构cv训练, 返回每个机构做oos下的train val oos ks, oos ks
    def train_epoch_(self, org, param):
        
        broadcast_with_tar = param.get('broadcast_with_tar')
        balanced_badrate = param.get("balanced_badrate")
        param['num_leaves'] = 2**param.get('max_depth')-1
        
        ## 根据超参数坏样率得出权重weight
        weight_tr = re_weight_by_org(self.y_tr, self.tr_orgidx, 0.5, broadcast_with_tar, balanced_badrate)
        weight_val = re_weight_by_org(self.y_val, self.val_orgidx, 0.5, broadcast_with_tar, balanced_badrate)
        weight = pd.concat([weight_tr, weight_val], axis=0)
        
        tr_idxs, val_idxs = set(self.X_tr.index), set(self.X_val.index)
        tr_idx, val_idx = self.tr_orgidx.get(org), self.val_orgidx.get(org)
        
        # 除去当前org选出训练验证集
        X_tr_, y_tr_ = self.X_tr.loc[list(tr_idxs-set(tr_idx)), ], self.y_tr.loc[list(tr_idxs-set(tr_idx)), ]
        X_val_, y_val_ = self.X_val.loc[list(val_idxs-set(val_idx)), ], self.y_val.loc[list(val_idxs-set(val_idx)), ]
        w_tr_, w_val_ = weight.loc[list(tr_idxs-set(tr_idx)), ], weight.loc[list(val_idxs-set(val_idx)), ]
#         w_tr_ = pd.Series(np.ones(X_tr_.shape[0]), index=X_tr_.index)
#         w_val_ = pd.Series(np.ones(X_val_.shape[0]), index=X_val_.index)
        
        # 去除的机构为oos
        X_oos, y_oos = pd.concat([self.X_tr.loc[tr_idx, ], self.X_val.loc[val_idx, ]], axis=0) , pd.concat([self.y_tr.loc[tr_idx, ], self.y_val.loc[val_idx, ]], axis=0)
        
        # 为了避免oos与val相同则两者weight长度相同进而无法分辨使用哪个权重计算val的 ks auc, 对y_oos做去一操作
        if len(y_val_) == len(y_oos):
            X_oos, y_oos = X_oos.iloc[1:], y_oos.iloc[1:]
        
        weights = {'train':w_tr_, 'val': w_val_, 'oos':pd.Series(np.ones(len(y_oos)))}
        
        ## 默认不需要评估，当早停存在使用metric中的auc评估，当record_train_process==True时开启自定义评估
        eval_results = {}
        valid_sets, valid_names, feval, callbacks = None, None, None, []
        train_set = lgb.Dataset(X_tr_, label=y_tr_, weight=w_tr_)
        val_set = lgb.Dataset(X_val_, label=y_val_, reference=train_set)
        oos_set = lgb.Dataset(X_oos, label=y_oos, reference=train_set)
        
        ## 判断是否需要早停，如果用户参数中给出了早停则固定最大迭代次数为300，否则不设置早停
        if 'stopping_rounds' in param.keys():
            param.update({'num_iterations': 300})
            callbacks.append(lgb.early_stopping(stopping_rounds=param.get('stopping_rounds')))
            ## 因为早停会每次迭代都评估 因为直接设为评估自定义函数
            self.record_train_process = True
            valid_sets = [train_set, val_set, oos_set]
            valid_names = ['train', 'val', 'oos']
        
        ## 判断是否开启自定义评估
        if self.record_train_process == True:
            callbacks.append(lgb.record_evaluation(eval_results))
            valid_sets = [train_set, val_set, oos_set]
            valid_names = ['train', 'val', 'oos']
            feval = self.weighted_metric(weights)
        
        # 训练模型, 评估训练 验证 oos
        ## valid_set若非空则中第一个非训练集的数据集是唯一决定早停的依据，评价方式是param中的metric, 与feval无关
        model = lgb.train(
                          param,
                          verbose_eval=0, 
                          train_set = train_set, 
                          valid_sets = valid_sets,
                          #  feval = ['auc', get_ks_func, get_lift_func],
                          #  eval_sample_weight = [w_tr_, w_val_, pd.series(np.ones(len(y_oos)))] 当切换lightgbm>=3.0.0可以使用
                          feval = feval,
                          valid_names = valid_names,
                          callbacks = callbacks
                         )
        
        if len(eval_results) > 0 and self.record_train_process == True:
            eval_results = pd.DataFrame(eval_results)
            eval_results['org'] = org
        
        if self.record_train_process == False:
            auc_w, auc, ks_w, ks, lift5, lift10 = self.single_weighted_metric(model, X_tr_, y_tr_, w_tr_)
            tmp0 = pd.DataFrame({'train':[[auc_w], [auc], [ks_w], [ks], [lift5], [lift10]]}, index=['auc_w', 'auc', 'ks_w', 'ks', '5%lift', '10%lift'])
            auc_w, auc, ks_w, ks, lift5, lift10 = self.single_weighted_metric(model, X_val_, y_val_, w_val_)
            tmp1 = pd.DataFrame({'val':[[auc_w], [auc], [ks_w], [ks], [lift5], [lift10]]}, index=['auc_w', 'auc', 'ks_w', 'ks', '5%lift', '10%lift'])
            auc_w, auc, ks_w, ks, lift5, lift10 = self.single_weighted_metric(model, X_oos, y_oos, pd.Series(np.ones(len(y_oos))))
            tmp2 = pd.DataFrame({'oos':[[auc_w], [auc], [ks_w], [ks], [lift5], [lift10]]}, index=['auc_w', 'auc', 'ks_w', 'ks', '5%lift', '10%lift'])
            eval_results = pd.concat([tmp0, tmp1, tmp2], axis=1)
            eval_results['org'] = org
        
        return eval_results

    # 自定义目标函数，当参数符合要求时进一步更新超参数空间寻优
    def objective(self, param):
        
        begin_time = time.time()
        if self.fobj is not None:
            param.update({'objective': self.fobj})
        
        results = pd.DataFrame()
        
        # 开启10个进程池运行lgb
        tasks = [(org, param) for org in self.tr_orgidx.keys()]
        with Pool(10) as pool:
            records = pool.starmap(self.train_epoch_, tasks)
        for record in records:
            results = pd.concat([results, record], axis=0)
        
        simpler_results = self.extract_evalresult(results)
        mean_val_ks, mean_oos_ks = np.mean(simpler_results['val_ks']), np.mean(simpler_results['oos_ks'])
        
        ## 防止内核关闭
        import json
        result = {'a':20}
        with open("res.json", 'w') as f:
            json.dump(result, f)
        
        # 判断参数符合要求条件为每个机构做oos时的训练集和验证集ks差距在相对3%以下，否则不更新loss
        if np.allclose(simpler_results['tr_auc'], simpler_results['val_auc'], rtol=self.auc_threshold) and np.allclose(simpler_results['tr_auc_w'], simpler_results['val_auc_w'], rtol=self.auc_threshold):
            loss = -mean_oos_ks
            status = STATUS_OK
        else:
            loss = np.Inf
            status = STATUS_FAIL
        
        end_time = time.time()
        display(f"当前组参数训练耗时：{np.round((end_time-begin_time)*1.0/60, 2)}分")
        
        return {'loss': loss, 'param':param, 'status':status, 'randn':self.randn,
                'mean_val_ks':mean_val_ks, 'mean_oos_ks':mean_oos_ks, 'simpler_results':simpler_results, 'results': results}
    
    # 该类的执行函数，返回trails
    def tpesearch_params(self):
        begin_time_ = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        display(f"开始执行时间：{begin_time_}")
        _ = fmin(fn=self.objective, space=self.params, algo=tpe.suggest, max_evals=self.max_iterations, trials=self.trails)
        return pd.DataFrame(self.trails)