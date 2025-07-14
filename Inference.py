from main.utils.data_augmentation import *
from main.utils.requirements import *
from main.utils.analysis import *

class Inference(object):
    '''
        init: data, oos_data, param, results, child_score, score_name, store_pth, randn, score_transform_func, model(optional)
        funcs: 
        
        return model, train log figures, model score figures, report(xlsx)
    '''
    def __init__(self, **kwargs):
        self.data = kwargs.get('data')
        self.oos_data = kwargs.get('oos_data')
        self.param = kwargs.get('param')
        self.param['num_leaves'] = 2**self.param.get('max_depth')-1
        self.results = kwargs.get("results")
        self.dataset_statis = kwargs.get("dataset_statis")
        self.child_score = kwargs.get("child_score")
        self.randn = kwargs.get('randn')
        self.score_name = kwargs.get('score_name')
        self.store_pth = kwargs.get("store_pth")
        self.score_transform_func = kwargs.get("score_transform_func")
        self.model = kwargs.get("model")
        self.feas_gain = None
        self.X_tr = None
        self.y_tr = None
        self.w_tr = None
        self.tr_orgidx = None
    
    def inference_split_data(self, data, flag):
        
        # 确定X，判定条件不是object类型且不是Y列
        feas = [v for v in data.columns if data[v].dtype!='O' and v!='new_target']
        
        # tr_orgidx存储训练集各个机构索引，tr_idx存储训练集全部索引, val为验证集同样做法
        tr_orgidx, val_orgidx, tr_idx, val_idx = {}, {}, [], []
        
        if flag==True:
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
            
            data_tr, data_val = data.loc[tr_idx, ], data.loc[val_idx, ]
            X_tr, X_val, y_tr, y_val = data_tr[feas], data_val[feas], data_tr['new_target'], data_val['new_target']
            return X_tr, X_val, y_tr, y_val, tr_orgidx, val_orgidx
           
        else:
            for org in data.new_org.unique():
                tmp_data = data[data.new_org==org].copy()
                org_index = tmp_data.index
                tr_orgidx[org] = list(org_index)
                tr_idx += list(org_index)
        
            data_tr = data.loc[tr_idx, ]
            X_tr, y_tr = data_tr[feas], data_tr['new_target']
            return X_tr, None, y_tr, None, tr_orgidx, None
    
    def _get_lift(self, y, pred, k):
        
        n_top = int(len(y) * k)
        top_indices = pd.Series(pred, index=y.index).sort_values(ascending=False).head(n_top).index
        
        return y[top_indices].mean() / y.mean()
    
    ## 为查看在n-1个机构上全量样本上训练的模型效果设置
    def inference_oos_metric(self, model, X, y):
        pred = model.predict(X)
        pred = pd.Series(pred, index=X.index)
        
        auc = roc_auc_score(y, pred)
        ks = toad.metrics.KS(pred, y)
        
        lift10, lift5, lift3 = self._get_lift(y, pred, 0.1), self._get_lift(y, pred, 0.05), self._get_lift(y, pred, 0.03)
        
        return  np.round(auc, 3), np.round(ks, 3), np.round(lift3, 2), np.round(lift5, 2), np.round(lift10, 2)
    
    ## 为子分生成评估metric
    def inference_childscore_metric(self, x, y):
        auc = roc_auc_score(y, x)
        ks = toad.metrics.KS(x, y)
        
        lift10, lift5, lift3 = self._get_lift(y, x, 0.1), self._get_lift(y, x, 0.05), self._get_lift(y, x, 0.03)
        
        return np.round(auc, 3), np.round(ks, 3), np.round(lift3, 2), np.round(lift5, 2), np.round(lift10, 2)
    
    ## 建模样本全量refit得到模型
    def refit(self):
        
        broadcast_with_tar = self.param.get('broadcast_with_tar')
        balanced_badrate = self.param.get("balanced_badrate")
        feas = [v for v in self.data.columns if self.data[v].dtype!='O' and v!='new_target']
        callbacks, val_set = [], None
        
        X_tr, X_val, y_tr, y_val, tr_orgidx, val_orgidx = self.inference_split_data(self.data, False)
        if balanced_badrate is not None:
            w_tr = re_weight_by_org(y_tr, tr_orgidx, 0.5, broadcast_with_tar, balanced_badrate)
        else:
            w_tr = pd.Series(np.ones(len(X_tr)), index=X_tr.index)
        train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        
        if 'stopping_rounds' in self.param.keys():
            self.param.update({'num_iterations': 300})
            callbacks.append(lgb.early_stopping(stopping_rounds=self.param.get('stopping_rounds')))
            
            # 更新训练集与其权重，添加验证集，早停判断标准和训练时一致为auc(不加权计算)
            X_tr, X_val, y_tr, y_val, tr_orgidx, val_orgidx = self.inference_split_data(self.data, True)
            
            if balanced_badrate is not None:
                w_tr = re_weight_by_org(y_tr, tr_orgidx, 0.5, broadcast_with_tar, balanced_badrate)
            else:
                w_tr = pd.Series(np.ones(len(X_tr)), index=X_tr.index)
            
            train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        
        self.model = lgb.train(
                          self.param,
                          verbose_eval=0, 
                          train_set = train_set, 
                          valid_sets = val_set,
                          callbacks = callbacks
                         )
        ## 保存模型
        joblib.dump(self.model, self.store_pth+self.score_name+'.pkl')
        
        return
    
    ## 基于建模样本单一机构做oos评估
    def refit_cvoos_(self, param, org, X_tr, y_tr, w_tr, tr_orgidx):
        print(org)
        tr_idxs = set(X_tr.index)
        oos_idx = set(tr_orgidx.get(org))
        X_tr_, y_tr_ = X_tr.loc[list(tr_idxs-oos_idx), ], y_tr.loc[list(tr_idxs-oos_idx), ]
        if param.get('balanced_badrate') is not None:
            w_tr_ = w_tr.loc[list(tr_idxs-oos_idx), ]
        else:
            w_tr_ = pd.Series(np.ones(X_tr_.shape[0]), index=X_tr_.index)
        train_set = lgb.Dataset(X_tr_, y_tr_, weight=w_tr_)
        X_oos, y_oos = X_tr.loc[list(oos_idx), ], y_tr.loc[list(oos_idx), ]

        cvoos_result = pd.DataFrame()

        ## n-1个机构全量样本下训练
        model = lgb.train(
                      param,
                      verbose_eval=0, 
                      train_set = train_set
                     )

        ## 得到oos结果
        auc, ks, lift3, lift5, lift10 = self.inference_oos_metric(model, X_oos, y_oos)
        nan_idx = list(X_oos[X_oos[self.child_score].isna()].index)
        auc_, ks_, lift3_, lift5_, lift10_ = self.inference_childscore_metric(X_oos[~X_oos.index.isin(nan_idx)][self.child_score], y_oos[~y_oos.index.isin(nan_idx)])
        cvoos_result = pd.concat([cvoos_result, pd.DataFrame({'oos': org, 'score': self.score_name, 'auc':auc, 'ks':ks, 
                                                '3%lift':lift3, '5%lift':lift5, '10%lift':lift10}, index=['0'])], axis=0)
        cvoos_result = pd.concat([cvoos_result, pd.DataFrame({'oos': org, 'score': self.child_score, 'auc':auc_, 'ks':ks_, 
                                                '3%lift':lift3_, '5%lift':lift5_, '10%lift':lift10_}, index=['0'])], axis=0)
        return cvoos_result
    
        
    ## 目前仅支持对不设置早停的参数生成oos结果
    def get_cvoos_result(self):
    
        assert 'stopping_rounds' not in self.param.keys(), "参数中存在stopping_rounds, 无法生成oos"
        broadcast_with_tar = self.param.get('broadcast_with_tar')
        balanced_badrate = self.param.get("balanced_badrate")
        
        X_tr, _, y_tr, _, tr_orgidx, _ = self.inference_split_data(self.data, False)
        if balanced_badrate is not None:
            w_tr = re_weight_by_org(y_tr, tr_orgidx, 0.5, broadcast_with_tar, balanced_badrate)
        else:
            w_tr = pd.Series(np.ones(X_tr.shape[0]), index=X_tr.index)
        
        cvoos_result = pd.DataFrame()
                # 每次1个机构做oos, 其余机构做trainset
        for org in self.data.new_org.unique():
            
            tr_idxs = set(X_tr.index)
            oos_idx = set(tr_orgidx.get(org))
            X_tr_, y_tr_ = X_tr.loc[list(tr_idxs-oos_idx), ], y_tr.loc[list(tr_idxs-oos_idx), ]
            if balanced_badrate is not None:
                w_tr_ = w_tr.loc[list(tr_idxs-oos_idx), ]
            else:
                w_tr_ = pd.Series(np.ones(len(X_tr_)), index=X_tr_.index)
            train_set = lgb.Dataset(X_tr_, y_tr_, weight=w_tr_)
            X_oos, y_oos = X_tr.loc[list(oos_idx), ], y_tr.loc[list(oos_idx), ]
            
            ## n-1个机构全量样本下训练
            model = lgb.train(
                          self.param,
                          verbose_eval=0, 
                          train_set = train_set
                         )
            
            ## 得到oos结果
            auc, ks, lift3, lift5, lift10 = self.inference_oos_metric(model, X_oos, y_oos)
            nan_idx = list(X_oos[X_oos[self.child_score].isna()].index)
            auc_, ks_, lift3_, lift5_, lift10_ = self.inference_childscore_metric(X_oos[~X_oos.index.isin(nan_idx)][self.child_score], y_oos[~y_oos.index.isin(nan_idx)])
            cvoos_result = pd.concat([cvoos_result, pd.DataFrame({'oos': org, 'score': self.score_name, 'auc':auc, 'ks':ks, 
                                                    '3%lift':lift3, '5%lift':lift5, '10%lift':lift10}, index=['0'])], axis=0)
            cvoos_result = pd.concat([cvoos_result, pd.DataFrame({'oos': org, 'score': self.child_score, 'auc':auc_, 'ks':ks_, 
                                                    '3%lift':lift3_, '5%lift':lift5_, '10%lift':lift10_}, index=['0'])], axis=0)
            
        return cvoos_result
        
#         # 开启10个进程池运行lgb
#         tasks = [(org) for org in self.tr_orgidx.keys()]
#         results = []
#         pool = multiprocessing.Pool()
#         for task in tasks:
#             results.append(pool.apply_async(self.refit_cvoos_, task))
#         pool.join()
#         pool.close()
#         with Pool(9) as pool:
#             records = pool.starmap(self.refit_cvoos_, tasks)
        
#         return cvoos_result
    
    def fixedbins_results(self, data, col, n_bins):
        
        tmp = data[[col, 'new_target']].copy()
        
        combiner = toad.transform.Combiner()
        combiner.fit(tmp, y=tmp['new_target'], method='quantile', n_bins=n_bins)
        bin_edges = combiner.export().get(col)
        min_edge = np.min(tmp[col])
        max_edge = np.max(tmp[col])
        
        def _bin_to_interval(bin_label, bin_edges):
            if pd.isna(bin_label):
                return 'NaN'
            bin_label = int(bin_label)
            if bin_label == 0:
                return f"[{min_edge}, {bin_edges[0]})"
            elif bin_label == len(bin_edges):
                return f"[{bin_edges[-1]}, {max_edge}]"
            else:
                return f"[{bin_edges[bin_label-1]}, {bin_edges[bin_label]})"

        tmp['bin_'] = combiner.transform(tmp[[col]])[[col]]
        tmp['bin'] = tmp['bin_'].apply(lambda x: _bin_to_interval(x, bin_edges))
        tmp['bin'] = tmp['bin'].astype(str)
        
        res = tmp.groupby(['bin']).apply(
            lambda x: pd.Series({
                '变量名': col,
                '分箱': x['bin'].iloc[0],
                '命中率': round(len(x)*1.0 / tmp.shape[0], 4),
                '坏样率': round(np.mean(x['new_target']), 4),
                '分箱正样本数': x['new_target'].sum(),
                '分箱负样本数': len(x) - x['new_target'].sum(),
                '分箱ks': round(toad.metrics.KS(x[col], x['new_target']), 4),
                 ## 默认使用等频5分箱计算每个x的iv值
                '分箱iv': calculate_iv(x[[col]], x['new_target'], 'quantile', 5)[0],
                '分箱lift': round(x['new_target'].mean() / tmp['new_target'].mean(), 2),
                '召回率': round(sum(x['new_target']) * 1.0 / sum(tmp['new_target']) ,4),
                '特异度': round((len(x)-sum(x['new_target'])) * 1.0 / (tmp.shape[0] - sum(tmp['new_target'])), 4)
            })
        )
        
        res = pd.DataFrame(res).reset_index(drop=True).sort_values(by=['分箱'])
        
        res['分箱iv'].replace({np.inf: 0}, inplace=True)
        res['总IV'] = sum(res['分箱iv'])
        res['累计正样本数'], res['累计负样本数'] =  res['分箱正样本数'].cumsum(), res['分箱负样本数'].cumsum()
        res['累计样本数'] = res['累计正样本数'] + res['累计负样本数']
        res['累积命中率'] = res['命中率'].cumsum()
        res['累积提升度'] = round((res['累计正样本数']/res['累计样本数'])*1.0/tmp['new_target'].mean(), 4)
        res['累积召回率'] = round(res['累计正样本数']*1.0/tmp['new_target'].sum(), 4)
        res['累积特异度'] = round(res['累计负样本数']*1.0/(tmp.shape[0]-tmp['new_target'].sum()), 4)
        
        return res
    
    def generate_report(self):
        
        if self.model is None:
            print("step 1 使用data拟合模型")
            self.refit()
        else:
            print("step 1 加载输入的模型")
        
        try:
            Path(self.store_pth+"/train logs/auc").mkdir(parents=True, exist_ok=True)
            Path(self.store_pth+"/train logs/ks").mkdir(parents=True, exist_ok=True)
            Path(self.store_pth+"/train logs/lift").mkdir(parents=True, exist_ok=True)
            Path(self.store_pth+"/train logs/trend").mkdir(parents=True, exist_ok=True)
            print(f"train log目录已就绪")
        except Exception as e:
            print(f'{e} train log目录生成失败')
        
        ## 得到train中每个机构做oos结果
        print("step2 计算cv oos结果")
        cv_trainoos_result = self.get_cvoos_result()
        
        ## 得到真正的oos set结果
        cv_oos_result = pd.DataFrame()
        for org in self.oos_data.new_org.unique():
            oos_data_ = self.oos_data[self.oos_data.new_org==org].copy()
            auc, ks, lift3, lift5, lift10 = self.inference_oos_metric(self.model, oos_data_[self.model.feature_name()], oos_data_['new_target'])
            cv_oos_result = pd.concat([cv_oos_result, pd.DataFrame({'oos': org, 'score':self.score_name, 'auc':auc, 'ks':ks, '3%lift':lift3, 
                                                    '5%lift':lift5, '10%lift':lift10}, index=['0'])], axis=0)
            ## 计算子分
            nan_idx = list(oos_data_[oos_data_[self.child_score].isna()].index)
            auc_, ks_, lift3_, lift5_, lift10_ = self.inference_childscore_metric(oos_data_[~oos_data_.index.isin(nan_idx)][self.child_score], oos_data_[~oos_data_.index.isin(nan_idx)]['new_target'])
            cv_oos_result = pd.concat([cv_oos_result, pd.DataFrame({'oos': org, 'score':self.child_score, 'auc':auc_, 'ks':ks_, '3%lift':lift3_, 
                                                    '5%lift':lift5_, '10%lift':lift10_}, index=['0'])], axis=0)
        
        cv_oos_result = pd.concat([cv_trainoos_result, cv_oos_result], axis=0)
        
        
        ## 保存gain值图
        self.feas_gain = pd.DataFrame(list(dict(zip(self.model.feature_name(), self.model.feature_importance(importance_type='gain'))).items()))
        self.feas_gain = self.feas_gain.sort_values(by=1, ascending=False)
        self.feas_gain['2'] = round(self.feas_gain[1] / sum(self.feas_gain[1]), 2)
        self.feas_gain.columns = ['变量', 'gain', 'gain值占比']
        
        ##绘制shap图
        #explain = shap.TreeExplainer(self.model)
        #shap_values = explain.shap_values(self.oot_data[self.model.booster_.feature_name()])
        #shap.summary_plot(shap_values[1], self.oot_data[self.model.booster_.feature_name()], max_display=20)
        #plt.savefig(self.store_pth+"shap_summary.jpg", dpi=300, bbox_inches="tight")
        #plt.close()
        
        self.data[self.score_name] = np.round(self.model.predict(self.data[self.model.feature_name()]), 3)
        self.oos_data[self.score_name] = np.round(self.model.predict(self.oos_data[self.model.feature_name()]), 3)
        
        if self.score_transform_func is not None:
            self.data[self.score_name]  = self.score_transform_func(self.data[self.score_name])
            self.oos_data[self.score_name]  = self.score_transform_func(self.oos_data[self.score_name])
        
        psi_oos = calculate_psi(self.data[self.score_name], self.oos_data[self.score_name])
        print(f"psi_oos是{psi_oos}")
        
        ## 绘制箱线图
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.boxplot(y='new_org', x=self.child_score, data=self.data, ax=ax1)
        fig1.savefig(self.store_pth+"/train logs/trend/子分箱线图.jpg")
        plt.close()
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(y='new_org', x=self.score_name, data=self.data, ax=ax2)
        fig2.savefig(self.store_pth+"/train logs/trend/评分箱线图.jpg")
        plt.close()
        
        print("step 3 计算模型得分&子分等频10分箱")
        bins_results = self.fixedbins_results(self.data, self.score_name, 10)
        bins_results_oos = self.fixedbins_results(self.oos_data, self.score_name, 10)
        bins_results_childscore = self.fixedbins_results(self.data, self.child_score, 10)
        bins_results_oos_childscore = self.fixedbins_results(self.oos_data, self.child_score, 10)
        
        print("step 4 分机构计算模型得分&子分等频10分箱")
        bins_results_org = pd.DataFrame()
        for org in self.data.new_org.unique():
            tmp_data = self.data[self.data.new_org==org].copy()
            try:
                bins_results_org_ = self.fixedbins_results(tmp_data, self.score_name, 10)
            except:
                print(f"{org}评分分为10箱中有部分箱全为0|1, 改为3分箱")
                bins_results_org_ = self.fixedbins_results(tmp_data, self.score_name, 3)
                
            bins_results_org_.insert(0, '机构', org)
            bins_results_org = pd.concat([bins_results_org, bins_results_org_], axis=0)
            
            try:
                bins_results_org_ = self.fixedbins_results(tmp_data, self.child_score, 10)
            except:
                print(f"{org}子分分为10箱中有部分箱全为0|1, 改为3分箱")
                bins_results_org_ = self.fixedbins_results(tmp_data, self.child_score, 3)
            bins_results_org_.insert(0, '机构', org)
            bins_results_org = pd.concat([bins_results_org, bins_results_org_], axis=0)
        
        
        print("step 5 生成评分趋势图")
        ## 保存data数据下得分10分箱子坏样率趋势图
        iv, edge_dict, _ = calculate_iv(self.data[self.score_name], self.data['new_target'], 'quantile', 10)
        iv, woe, fig = trend_detect(self.data[self.score_name], self.data['new_target'], edge_dict)
        fig.savefig(self.store_pth+"/train logs/trend/建模样本得分趋势图.jpg")
        ## 保存oos_data数据下得分10分箱子坏样率趋势图
        iv, edge_dict, _ = calculate_iv(self.oos_data[self.score_name], self.oos_data['new_target'], 'quantile', 10)
        iv, woe, fig = trend_detect(self.oos_data[self.score_name], self.oos_data['new_target'], edge_dict)
        fig.savefig(self.store_pth+"/train logs/trend/oos得分趋势图.jpg")
        
        print("step 6 生成训练过程图")
        for org in self.results.org.unique():
    
            results_auc = self.results[(self.results.org==org) & (self.results.idx=='auc')]
            results_auc_w = self.results[(self.results.org==org) & (self.results.idx=='auc_w')]
            results_ks = self.results[(self.results.org==org) & (self.results.idx=='ks')]
            results_ks_w = self.results[(self.results.org==org) & (self.results.idx=='ks_w')]
            results_5lift = self.results[(self.results.org==org) & (self.results.idx=='5%lift')]
            results_10lift = self.results[(self.results.org==org) & (self.results.idx=='10%lift')]

            x = np.arange(len(ast.literal_eval(results_auc.train.iloc[0])))
            tr_auc, val_auc, oos_auc = ast.literal_eval(results_auc.train.iloc[0]), ast.literal_eval(results_auc.val.iloc[0]), ast.literal_eval(results_auc.oos.iloc[0])
            tr_auc_w, val_auc_w, oos_auc_w = ast.literal_eval(results_auc_w.train.iloc[0]), ast.literal_eval(results_auc_w.val.iloc[0]), ast.literal_eval(results_auc_w.oos.iloc[0])
            df = pd.DataFrame({
                    'iteration': np.tile(x, 6),
                    'value': np.concatenate([tr_auc, val_auc, oos_auc, tr_auc_w, val_auc_w, oos_auc_w]),
                    'type': ['train']*len(x) + ['val']*len(x) + ['oos']*len(x) + ['train_w']*len(x) + ['val_w']*len(x) + ['oos_w']*len(x),
                    'weight': ['no']*3*len(x) + ['yes']*3*len(x)
                })

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
            sns.lineplot(
                data=df[df.weight=='no'],
                x='iteration',
                y='value',
                hue='type',
                ax=ax1
            )
            ax1.set_title(f"auc, 贷外是{org}")

            sns.lineplot(
                data=df[df.weight=='yes'],
                x='iteration',
                y='value',
                hue='type',
                ax=ax2
            )
            ax2.set_title(f"加权auc, 贷外是{org}")
            plt.tight_layout()
            fig.savefig(f'{self.store_pth}/train logs/auc/auc_{org}.jpg')

            ## 绘制ks图
            tr_ks, val_ks, oos_ks = ast.literal_eval(results_ks.train.iloc[0]), ast.literal_eval(results_ks.val.iloc[0]), ast.literal_eval(results_ks.oos.iloc[0])
            tr_ks_w, val_ks_w, oos_ks_w = ast.literal_eval(results_ks_w.train.iloc[0]), ast.literal_eval(results_ks_w.val.iloc[0]), ast.literal_eval(results_ks_w.oos.iloc[0])
            df = pd.DataFrame({
                    'iteration': np.tile(x, 6),
                    'value': np.concatenate([tr_ks, val_ks, oos_ks, tr_ks_w, val_ks_w, oos_ks_w]),
                    'type': ['train']*len(x) + ['val']*len(x) + ['oos']*len(x) + ['train_w']*len(x) + ['val_w']*len(x) + ['oos_w']*len(x),
                    'weight': ['no']*3*len(x) + ['yes']*3*len(x)
                })

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
            sns.lineplot(
                data=df[df.weight=='no'],
                x='iteration',
                y='value',
                hue='type',
                ax=ax1
            )
            ax1.set_title(f"ks, 贷外是{org}")

            sns.lineplot(
                data=df[df.weight=='yes'],
                x='iteration',
                y='value',
                hue='type',
                ax=ax2
            )
            ax2.set_title(f"加权ks, 贷外是{org}")
            plt.tight_layout()
            fig.savefig(f'{self.store_pth}/train logs/ks/ks_{org}.jpg')

            ## 绘制lift
            tr_5lift, val_5lift, oos_5lift = ast.literal_eval(results_5lift.train.iloc[0]), ast.literal_eval(results_5lift.val.iloc[0]), ast.literal_eval(results_5lift.oos.iloc[0])
            tr_10lift, val_10lift, oos_10lift = ast.literal_eval(results_10lift.train.iloc[0]), ast.literal_eval(results_10lift.val.iloc[0]), ast.literal_eval(results_10lift.oos.iloc[0])
            df = pd.DataFrame({
                    'iteration': np.tile(x, 6),
                    'value': np.concatenate([tr_5lift, val_5lift, oos_5lift, tr_10lift, val_10lift, oos_10lift]),
                    'type': ['train']*len(x) + ['val']*len(x) + ['oos']*len(x) + ['train']*len(x) + ['val']*len(x) + ['oos']*len(x),
                    'weight': ['no']*3*len(x) + ['yes']*3*len(x)
                })

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
            sns.lineplot(
                data=df[df.weight=='no'],
                x='iteration',
                y='value',
                hue='type',
                ax=ax1
            )
            ax1.set_title(f"5%lift, 贷外是{org}")

            sns.lineplot(
                data=df[df.weight=='yes'],
                x='iteration',
                y='value',
                hue='type',
                ax=ax2
            )
            ax2.set_title(f"10%lift, 贷外是{org}")
            plt.tight_layout()
            fig.savefig(f'{self.store_pth}/train logs/lift/lift_{org}.jpg')
        
        print("step 7 生成模型报告中")
        writer = pd.ExcelWriter(self.store_pth+"/"+self.score_name+'.xlsx', engine='xlsxwriter')
        ## 表一
        self.dataset_statis.to_excel(writer, sheet_name='全量样本分机构概览', index=False)
        ## 表二
        cv_oos_result.to_excel(writer, sheet_name='分机构贷外效果', index=False)
        ## 表三
        bins_results.to_excel(writer, sheet_name='建模样本评分10分箱', index=False)
        ## 表四
        bins_results_oos.to_excel(writer, sheet_name='oos评分10分箱', index=False)
        ## 表五
        bins_results_childscore.to_excel(writer, sheet_name='建模样本子分10分箱', index=False)
        ## 表六
        bins_results_oos_childscore.to_excel(writer, sheet_name='oos子分10分箱', index=False)
        ## 表七
        bins_results_org.to_excel(writer, sheet_name='分机构评分&子分10分箱', index=False)
        ## 表八
        self.feas_gain.to_excel(writer, sheet_name='模型gain值排序', index=False)
        ## 表九
        self.param['randn'] = self.randn
#         self.param['train orgs'] = list(self.data.new_org.unique())
#         self.param['oos orgs'] = list(self.oos_data.new_org.unique())
        pd.DataFrame(self.param, index=['0']).to_excel(writer, sheet_name='模型参数', index=False)

        writer.save()
        writer.close()
        
        print("已完成模型报告")
        return