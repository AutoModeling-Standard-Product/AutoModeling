b# 随机种子管理
random_states = {
    'spliter': 126,        # 数据集划分随机种子
    'resampler': 252,      # 采样器随机种子
    'trainer': 42,         # 训练器随机种子
    'optimizer': 84        # 优化器随机种子
}



# 数据读取参数，这里假设X与y同在一张表中，否则需要另行预先处理
reader_ = {
    'data_pth':'modeling_data.csv',                          # 数据路径
    'date_colName': 'apply_date',                            # 指定日期标识列名
    'y_colName': 'credit_target',                            # 指定目标变量列名
    'org_colName':'org',                                     # 指定机构标识列名
    'data_encode':'utf-8',                                   # 编码方式
    'key_colNames' : ['mobile_org', 'apply_date', 'org']     # 主键字段
}

# 数据清洗与筛选阈值
filter_ = {
    'minYmBadsample': 10,        # 单机构每月坏样本数量下限
    'minYmSample': 1000,         # 单机构每月总样本数量下限
    'max_missing_value': 0.9,    # 特征缺失率上限阈值
    'min_iv': 0.01,              # 特征IV下限阈值
    'max_psi': 0.1               # 特征PSI上限阈值
}

# 数据划分参数
spliter_ = {
    'oot_size': 0.1,        # oot划分比例(采用qcut截取后段)，默认应用于各机构划分，也可以应用于总体样本的一次性划分
    'train_size': 0.8       # 训练集划分比例，用于初始划分
}

# 样本均衡参数，可直接设置为None以跳过采样器
resampler_ = None
# resampler_ = {
#     'type': 'SMOTE',                                     # 采样器名称                 
#     'params': {
#         'sampling_strategy': 'auto', 
#         'random_state': random_states['resampler'],      # 采样器参数 
#         'k_neighbors': 5
#     }
# }

# 建模参数
classifier_ = {
    'type': lgb,                          # 选择建模算法，目前默认lgb
    'params': {                           # 这里给出一些固定的，不随调参过程而变动的参数
        'objective': 'binary',            
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': 2,
        'num_threads': 3,
        'seed': random_states['trainer']
    },               
    'scale_pos_weight': None,    # None表示不调整模型自带的权重平衡，否则给出能计算得到权重（浮点数）的函数，并更新参数字典params
    'weight': None,           # None表示权重参数默认为np.ones，否则给出能计算得到权重序列的函数，并更新参数字典params
    'fobj': None,                         # None表示不调整目标函数，否则给出新的目标函数，并更新参数字典params
    'feval': None                         # None表示不调整评价函数，否则给出新的评价函数，并更新参数字典params
}

# 优化器参数
optimizer_ = {
    'type': 'bayes',            # 这里直接填写优化器名称（字符串），可选'grid'和'bayes'
#     'pbounds': {
#         'num_iterations': np.arange(80, 100, 3),
#         'learning_rate':[0.05],
#         'colsample_bytree': [0.6],
#         'max_depth': [4, 5],
#         'max_bin': np.arange(50, 100, 10),
#         'min_child_weight': [25],
#         'reg_alpha': [3],
#         'reg_lambda': [1], 
#     },
    'pbounds':{
        'num_iterations': scope.int(hp.quniform('num_iterations',70, 150, 5)), 
        'learning_rate':hp.quniform('learning_rate', 0.01, 0.05, 0.01),
        'colsample_bytree':hp.quniform('colsample_bytree',0.5, 0.9, 0.1),
        'max_depth': scope.int(hp.quniform('max_depth',3, 7, 1)), 
        'max_bin': scope.int(hp.quniform('max_bin',50, 150, 10)), 
        'min_child_weight': scope.int(hp.quniform('min_child_weight', 10, 30, 5)),
        'reg_alpha': hp.quniform('reg_alpha', 1, 10, 1),
        'reg_lambda': hp.quniform('reg_lambda', 1, 10, 1),
    },                         # 参数搜索空间，网格调参需给出列表，贝叶斯调参需给出范围元组
    'max_iter': 100,
    'max_gap': 3e-2            # 优化目标约束，单机构训练集与oos在ks上的表现差距不能超过该约束
}


# 业务参数
channel = {'银行': ['字节放心借', '滴滴金融'], '非银': ['xk_hc','fd_hc', '长银消金', '久恒融担（恒昌）', '分期乐欺诈', '宁夏海胜通']}
channel.update({'整体': channel['银行'] + channel['非银']})

# 模型报告路径
report_path = 'report.xlsx'


# 数据增强约束校验（可选）：全流程中至多打开一个与数据增强相关的开关，防止多个增强过程相互干扰
imb_restrict = True
if imb_restrict:
    assert sum([
        resampler_ is not None, 
        classifier_['weight'] is not None,
        classifier_['scale_pos_weight'] is not None,
        classifier_['fobj'] is not None or classifier_['feval'] is not None
    ]) <= 1, f'已开启的数据增强开关超过1个，参数设置不合格'