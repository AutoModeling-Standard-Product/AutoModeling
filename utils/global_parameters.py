from utils.requirements import *

params4eda = {
    'data_pth': '', ## 数据集路径，最好给绝对路径
    'store_pth': '', ## eda结果的输出路径
    'oos_orgs': [], ## oos机构
    
    'date_colName': 'apply_date', ## 日期列列名
    'y_colName': 'target', ## 标签列列名
    'org_colName': 'org', ## 可选，机构列列名，为None时生成唯一机构列标识
    'data_encode': 'utf-8', ## 读取数据的解码'GB2312', 'gbk', 'utf-8'
    'key_colNames' : ['matchingid'], ## 主键列，去重用
    'drop_colNames': [], ## 指定读取文件时去除的列
    'minYmBadsample': 5, ##建议5
    'minYmSample': 0, ## 建议500
    
    'channels': {'银行': [],
           
            '非银24': [],
                
            '非银36':[],
            
           '整体': []},
    
    'missing_ratio': 0.6, ## 建议0.6
    'missing_orgcnt': 5, ## 建议总机构数的1/3
    'bin_method': 'quantile', ## 可选['quantile', 'dt'], 建议quantile
    'bin_cnt': 5, ## 建议5
    'miniv_org': 0.01, ## 建议0.01
    'miniv_channel': 0.01, ## 建议0.05
    'lowiv_orgcnt': 20, ## 建议总机构数的2/3
    'highpsi_orgcntnt': 15, ## 建议总机构数的1/3
    'max_psi': 0.1, ## 建议0.1
    'corr_method': 'pearson', ## 可选['spearman', 'pearson'], 建议spearman
    'max_corr': 0.85 ## 建议0.85
}

paramspace = {
          'num_threads': 1, ## 请不要修改进程数,每个lgb.train()只允许使用单进程, 否则每组参数下的多进程无法执行
          'boosting_type': hp.choice('boosting_type', ['gbdt']),
          'num_iterations': scope.int(hp.quniform('num_iterations',200, 520, 10)), 
          'learning_rate':hp.quniform('learning_rate', 0.04, 0.07, 0.01),
          'colsample_bytree':hp.quniform('colsample_bytree',0.5, 0.95, 0.05),
          'max_depth': scope.int(hp.quniform('max_depth',3, 7, 1)), 
          'max_bin': scope.int(hp.quniform('max_bin',80, 200, 20)), 
          'min_child_samples': scope.int(hp.quniform('min_child_samples', 1000, 10000, 1000)),
          'reg_alpha': hp.quniform('reg_alpha', 1, 100, 5),
          'reg_lambda': hp.quniform('reg_lambda', 1, 100, 5),
          'balanced_badrate': hp.quniform('balanced_badrate', 5, 15, 2),
          'broadcast_with_tar': hp.choice('broadcast_with_tar', [False]), # 如果设置了balanced_badrate不为空 则参数不起作用
          'objective':'binary',
          'metric':'auc' # 请不要去掉此参数，如果有早停将作为依据
}

params4hyperopt = {'params': paramspace,
                   'fobj':None, ## 自定义损失函数
                   'max_iterations': 5, ## 最大寻优次数, 建议500
                   'auc_threshold':5e-2, ## 相对gap, 建议为0.05
                   'randn':42 ## 训练集验证机切分随机数种子
}

params4inference = {'score_name': None,## 模型评分自定义名称
                    'child_score':None, ## 入模子分，为了模型效果比对使用，暂时只支持输入单个
                    'score_transform_func':None ## 分数转换函数
}