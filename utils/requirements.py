##-- package dependences
## how to pip: !sudo /opt/conda3/bin/pip install -i http://192.168.101.40/pypi/simple --trusted-host 192.168.101.40 toad==0.0.64 imbalanced-learn==0.6.2 scikit-learn==0.24.2 shap==0.36.0 numba==0.48 llvmlite==0.31.0 mxnet==1.4.0 pathos
import pandas as pd
import numpy as np
import pickle
import tqdm
from typing import Tuple
import toad
import shap
import joblib
from joblib import load
import seaborn as sns
import matplotlib.pyplot as plt
from toad.plot import bin_plot, badrate_plot
from datetime import datetime
from dateutil.relativedelta import relativedelta
import itertools
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, ParameterSampler
from sklearn.metrics import roc_auc_score,roc_curve, auc,precision_recall_curve,average_precision_score,confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import *
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, Trials,space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope
from scipy.stats import ks_2samp
import time
from functools import partial
import multiprocessing as multiprocessing
from multiprocessing import Pool, Manager, connection
# import utils.multiprocessing as multiprocessing
# from utils.multiprocessing import Pool, Manager, connection
import ast
import os
from pathlib import Path