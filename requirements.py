##-- package dependences
## how to pip: !sudo /opt/conda3/bin/pip install -i http://192.168.101.40/pypi/simple --trusted-host 192.168.101.40 packageyouneeded
import pandas as pd
import numpy as np
import pickle
import tqdm
from typing import Tuple
import toad
import matplotlib.pyplot as plt
from toad.plot import bin_plot, badrate_plot
from datetime import datetime
from dateutil.relativedelta import relativedelta
import itertools
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, ParameterSampler
from sklearn.metrics import roc_auc_score,roc_curve, auc,precision_recall_curve,average_precision_score,confusion_matrix
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, Trials,space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope
from scipy.stats import ks_2samp
import time
from functools import partial
from multiprocessing import Pool
import shap