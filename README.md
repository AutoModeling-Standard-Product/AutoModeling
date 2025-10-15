# AutoModeling

| 文件目录              | 文件用途                                        |
| -------------------- | ----------------------------------------------- |
| multiprocessing      | 多进程库（修复python3.6及以下版本的多进程通信容量问题）  |
| auto_modeling.ipynb  | notebook执行文件           |
| pip_list.txt         | 依赖包列表                                  |
| func.py              | 数据概览与清洗、变量筛选等基础函数              |
| analysis.py          | 提供集成func.py的函数供用户调用从而完成功能     |
| data_augmentation.py | 提供数据过采样及样本赋训练权重的方法            |
| global_parameters.py | 全流程的预设参数，请在执行notebook前先编写好该部分 |
| hyperopt_lgb.py      | 基于贝叶斯优化的机构留一法调参     |
| Inference.py | 最优参数模型拟合及模型报告生成 |
| requirements.py | 导入库 |

# 使用流程：
  1. 请预先手动设置global_parameters.py内的所有参数。
     - params4eda参数channel请保留渠道“整体”；
     - paramspace和params4hyperopt为建模调参参数，可直接沿用或根据数据情况自定义各指标值，建议不要修改指标结构；
     - 如果有需要查看某些子分或重要变量的分箱情况，请设置params4inference中的child_score(当前脚本只支持一个child_score的设置)。
  2. 依次执行auto_modeling.ipynb的各单元格：
     - 导入所需库及导入执行类auto_ML；
     - 创建对象auto_ml并执行数据清洗筛选，如不需要相关性及高噪声筛选，请注释step 5及step 6；
     - 开始调参（finetuingParams）
     - 创建报告（auto_ml.inference_model）

     
## 以下为数据筛选环节的函数简介，供参考

## func.py

### get_dataset(**kwargs)

- 功能：获取格式化后的数据集

- 输入：超参数`kwargs={'data_pth':String,'date_colName': String, 'y_colName':String, 'org_colName':String, 'data_encode':String, 'key_colName':List)}`. `data_pth`为数据集绝对路径(仅支持`csv, xlsx, pkl`)，`date_colName`为日期列名，`y_colName`为标签列名，`org_colName`为机构列名（若为空则自动生成），`data_encode`为数据集解码方式，`key_colName`为主键名.

- 输出：`pandas.DataFrame`, 其中标签列重名为`new_target`, 日期列格式为年月日(`new_date`)与年月(`new_date_ym`)两列, 机构列重命名为`new_org`, 数据根据主键`key_colName`去重.


### org_analysis(data)

- 功能：提供对机构层面下的坏样本|坏样率|样本数信息概览

- 输入:`pandas.DataFrame`

- 输出：`pandas.DataFrame`, 内容为每个机构每个月份下的坏样本数，样本数，坏样本率；每个机构下的总坏样本数，坏样率




### miss_check(**kwargs)

- 功能：提供对渠道|机构层面变量的-1111|-999|-1|nan|总缺失率

- 输入：超参数`kwargs={'data':DataFrame, 'channel': dict}`.`channel`允许输入多个渠道，不同渠道内的机构可以重合

- 输出：`Tuple[pandas.DataFrame，pandas.DataFrame]`, 每个变量在各个机构下的缺失率情况，在各个渠道下的缺失率情况




### calculate_psi(base, current)

- 功能：计算等频10箱下的psi值

- 输入：`base: pd.Series, current: pd.Series`

- 输出：`float64`,保留四位有效数字




### detect_psi(**kwargs)

- 功能：等频4分箱计算机构|渠道层面下的变量psi

- 输入：超参数 `kwargs={'data':DataFrame,'channel':dict}`

- 输出：`Tuple[pandas.DataFrame, pandas.DataFrame]`, 变量在机构|渠道下的等频4分箱样本数 分箱内psi 单机构下最大psi.




### calculate_iv(x, y, method, bins)

- 功能：计算指定分箱方法 分箱数下的变量iv值

- 输入：`x:pd.Series, y:pd.Series, method:String, bins: int`, `y`只允许是0或1， `method`只允许是`dt`或`quantile`, `bins`必须>=2

- 输出：`Tuple[float64, dict, pd.Series]`, iv值 分箱边界(toad.export原生)  分箱后`x`




### detect_iv(**kwargs)

- 功能：计算机构|渠道层面下变量的iv

- 输入：超参数 `kwargs={'data':DataFrame, 'channel':dict, 'method':String, 'bins': int}`,`y`只允许是0或1， `method`只允许是`dt`或`quantile`, `bins`必须>=2

- 输出：`Tuple[pandas.DataFrame, pandas.DataFrame]`, 变量在机构|渠道下的iv值




### trend_detect(x, y, edge_dict)

- 功能：给定分箱边界绘制变量坏样率|样本数趋势图并返回`iv`值，`woe`转换后的`x`

- 输入：`x:pd.Series, y:pd.Series, edge_dict: dict`

- 输出： `Tuple[jpg, float64, pd.Series]`, 变量指定分箱下的坏样率|样本数图例，变量`iv`值，`woe`转换后的`x`




### detect_correlation(**kwargs)

- 功能：计算变量之间的相关性并返回与标签列new_target相似性大于0.5的变量以及高相关性的变量

- 输入：超参数`kwargs={'data':pandas.DataFrame, 'method':String, 'max_corr':flaot64}`, method只允许是pearson或spearman, max_corr属于0到1的开区间

- 输出：`Tuple[List, pandas.DataFrame]`, 相似性大于给定阈值的变量对，不包含标签列的相关性矩阵




### get_fixed_lgb(max_depth, n_estimator)

- 功能：获取给定树木深与数棵数的简易lgb二分类器

- 输入：`max_depth, n_estimator`

- 输出：`lightgbm.LGBMClassifier`




## analysis.py

### drop_abnormal_ym(**kwargs)

- 功能：去除数据中每个机构下坏样率或样本数异常的月份对应数据

- 输入：超参数`kwargs={'data':pandas.DataFrame, 'minYmBadsample':int, 'minYmSample': int}`, `minYmBadsample`是机构层面下每个月最小允许的坏样本数，`minYmSample`是机构层面下每月允许的最小样本数

- 输出：`pandas.DataFrame`




### drop_highmiss_features(*kwargs)

- 功能：去除在多个机构或整体下高缺失率的变量

- 输入：超参数`kwargs={'data': pandas.DataFrame, 'miss_org': pandas.DataFrame, 'miss_channel': pandas.DataFrame, 'ratio': float64, 'cnt': int}`, `miss_org`与`miss_channel`是通过函数`miss_check()`得到的结果，`ratio`是变量在单机构或渠道上允许的最大缺失率，`cnt`是变量在设定最大缺失率下的允许出现的最大机构次数S

- 输出：`pandas.DataFrame`




### drop_lowiv_features(**kwargs)

- 功能：去除在多个机构上且任意渠道上低iv的变量，当变量在多个机构上低于iv阈值但任一渠道iv不低于阈值则保留

- 输入：超参数`kwargs={'data':pandas.DataFrame, 'res_iv_org':pandas.DataFrame, 'res_iv_channel': pandas.DataFrame, 'miniv':float64}`, `res_iv_org`与`res_iv_channel`是函数`detect_iv()`得到的结果, `miniv`是变量在单机构或渠道上允许的最小`iv`值

- 输出：`pandas.DataFrame`




### drop_highpsi_features(**kwargs)

- 功能：去除在多个机构上存在高psi的变量

- 输入：超参数`kwargs={'data':pandas.DataFrame, 'res_psi_org': pandas.DataFrame, 'ratio': float64, 'cnt': int}`, `res_psi_org`是函数`detect_psi()`得到的结果1即关于机构层面下变量`psi`的情况，`ratio`是变量在单机构上允许的最大`psi`，`cnt`是变量超过给定`psi`阈值(`ratio`)下最大允许出现的机构数

- 输出：`pandas.DataFrame`




### drop_highcorrelation_features(**kwargs)

- 功能：去除高度相关性的变量，根据给定渠道下高相似性变量的最小iv值对应变量进而去除高相似变量对，循环上述过程直至高相似变量对被全部去除

- 输入：超参数`kwargs={'data':pandas.DataFrame, 'res_iv_channel':pandas.DataFrame, 'indices': List, 'channel': String}`, `res_iv_channel`是函数`detect_iv()`下得到的结果2(渠道下的变量`iv`)，`indices`是函数`detect_correlation``()`返回的结果1(高度相关的变量对), `channel`是具体的渠道(属于函数`detect_iv()`的输入`channel.key()`的值)

- 输出：`pandas.DataFrame`




### drop_highnoise_features(**kwargs)

- 功能：使用null importance去除高噪音的变量，认为转变y前后变量的gain值差异在50之内即为高高噪音变量

- 输入：超参数`kwargs={'data':pandas.DataFrame, 'max_depth':int, 'n_estimator': int}`, `max_depth`与`n_estimator`是用于生成`lgb`分类器的输入

- 输出：`pandas.DataFrame`
