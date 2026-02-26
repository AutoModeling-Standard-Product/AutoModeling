"""
Optuna-based LightGBM Hyperparameter Optimization Module.

This module provides parameter tuning for LightGBM models using Optuna framework,
with Leave-One-Institution-Out (LOIO) cross-validation for risk modeling.

Key Features:
- Optuna-based hyperparameter optimization (replaces legacy hyperopt)
- LOIO cross-validation for institutional stability
- Weighted metrics (AUC, KS, Lift) for imbalanced data
- Stability checks to prevent overfitting
- Parallel training with multiprocessing

Author: Risk Modeling Team
Created: 2024
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
import time
import json
import multiprocessing as mp
from multiprocessing import Manager, Pool
import warnings

from ai_utils.logger import get_logger
from ai_utils.data_augmentation import re_weight_by_org

# Get logger for this module
logger = get_logger(__name__)


def _get_lift(y: pd.Series, pred: pd.Series, k: float) -> float:
    """
    Calculate Lift metric at top k percent.
    
    Args:
        y: True labels
        pred: Predicted probabilities
        k: Top k percent (e.g., 0.1 for top 10%)
    
    Returns:
        Lift value (top-k bad rate / overall bad rate)
    """
    n_top = int(len(y) * k)
    if n_top == 0:
        return 0.0
    
    top_indices = pred.sort_values(ascending=False).head(n_top).index
    top_bad_rate = y.loc[top_indices].mean()
    overall_bad_rate = y.mean()
    
    if overall_bad_rate == 0:
        return 0.0
    
    return top_bad_rate / overall_bad_rate


def _calculate_weighted_metrics(
    y: pd.Series, 
    pred: np.ndarray, 
    sample_weight: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Calculate both weighted and unweighted metrics (AUC, KS, Lift).
    
    Args:
        y: True labels
        pred: Predicted probabilities
        sample_weight: Optional sample weights
    
    Returns:
        Dictionary containing all metrics
    """
    pred_series = pd.Series(pred, index=y.index)
    
    # Unweighted metrics
    auc = roc_auc_score(y, pred_series)
    
    # Weighted metrics
    if sample_weight is not None and len(sample_weight) > 0:
        auc_w = roc_auc_score(y, pred_series, sample_weight=sample_weight)
        fpr, tpr, _ = roc_curve(y, pred_series, sample_weight=sample_weight)
        ks_w = float(max(tpr - fpr))
    else:
        auc_w = auc
        ks_w = 0.0
    
    # KS (unweighted, using toad-like calculation)
    fpr, tpr, _ = roc_curve(y, pred_series)
    ks = float(max(tpr - fpr))
    
    # Lift metrics
    lift_10 = _get_lift(y, pred_series, 0.1)
    lift_5 = _get_lift(y, pred_series, 0.05)
    
    return {
        'auc': auc,
        'auc_w': auc_w,
        'ks': ks,
        'ks_w': ks_w,
        'lift_10': lift_10,
        'lift_5': lift_5
    }


class OptunaLGB:
    """
    LightGBM Hyperparameter Optimizer using Optuna.
    
    This class implements Leave-One-Institution-Out (LOIO) cross-validation
    for hyperparameter tuning, ensuring model stability across institutions.
    
    Attributes:
        data: Input dataframe with features, target (new_target), and org (new_org)
        params: Base parameters for LightGBM
        max_iterations: Maximum number of Optuna trials
        n_jobs: Number of parallel jobs for institution training
    
    Example:
        >>> optimizer = OptunaLGB(
        ...     data=df,
        ...     params=base_params,
        ...     max_iterations=50,
        ...     n_jobs=4
        ... )
        >>> best_params = optimizer.optimize()
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        max_iterations: int = 50,
        n_jobs: int = 4,
        randn: int = 42,
        auc_threshold: float = 0.03,
        ks_threshold: float = 0.05,
        record_train_process: bool = False,
        early_stopping_rounds: int = 50,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        direction: str = 'maximize'
    ):
        """
        Initialize OptunaLGB optimizer.
        
        Args:
            data: DataFrame with features, 'new_target' (label), 'new_org' (institution)
            params: Base LightGBM parameters (search space will be built on top)
            max_iterations: Maximum number of Optuna trials
            n_jobs: Number of parallel processes for LOIO training
            randn: Random seed for reproducibility
            auc_threshold: Maximum allowed relative AUC difference (train vs val)
            ks_threshold: Maximum allowed relative KS difference (train vs val)
            record_train_process: Whether to record training metrics per iteration
            early_stopping_rounds: Early stopping rounds for LightGBM
            study_name: Name for Optuna study (for persistence)
            storage: Database URL for study storage (e.g., 'sqlite:///study.db')
            direction: Optimization direction ('maximize' or 'minimize')
        
        Raises:
            ValueError: If required columns missing or parameters invalid
        """
        # Validate input
        required_cols = ['new_target', 'new_org']
        missing_cols = [c for c in required_cols if c not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if data['new_org'].nunique() < 2:
            raise ValueError("Need at least 2 institutions for LOIO validation")
        
        self.data = data
        self.base_params = params.copy()
        self.max_iterations = max_iterations
        self.n_jobs = min(n_jobs, mp.cpu_count())
        self.randn = randn
        self.auc_threshold = auc_threshold
        self.ks_threshold = ks_threshold
        self.record_train_process = record_train_process
        self.early_stopping_rounds = early_stopping_rounds
        self.direction = direction
        
        # Split data: 80% train, 20% validation (stratified by institution)
        self.X_tr, self.X_val, self.y_tr, self.y_val, self.tr_orgidx, self.val_orgidx = \
            self._split_data()
        
        # Get feature columns
        self.feature_cols = [v for v in data.columns 
                            if data[v].dtype != 'O' and v not in ['new_target', 'new_org']]
        
        # Initialize Optuna study
        sampler = TPESampler(seed=randn)
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            direction=direction,
            load_if_exists=True
        )
        
        logger.info(f"Initialized OptunaLGB with {len(self.feature_cols)} features, "
                   f"{len(self.tr_orgidx)} institutions")
    
    def _split_data(self) -> Tuple:
        """
        Split data into train/validation sets with stratified sampling per institution.
        
        Returns:
            Tuple of (X_tr, X_val, y_tr, y_val, tr_orgidx, val_orgidx)
        """
        feas = self.feature_cols
        tr_orgidx, val_orgidx = {}, {}
        tr_idx, val_idx = [], []
        
        # Stratified split per institution
        splitter = StratifiedShuffleSplit(
            n_splits=1, 
            random_state=self.randn, 
            train_size=0.8
        )
        
        for org in self.data['new_org'].unique():
            tmp_data = self.data[self.data['new_org'] == org].copy()
            org_index = tmp_data.index.tolist()
            
            for idx_tr, idx_val in splitter.split(
                tmp_data[feas], 
                tmp_data['new_target']
            ):
                tr_orgidx[org] = [org_index[i] for i in idx_tr]
                val_orgidx[org] = [org_index[i] for i in idx_val]
                val_idx.extend([org_index[i] for i in idx_val])
                tr_idx.extend([org_index[i] for i in idx_tr])
        
        X_tr = self.data.loc[tr_idx, feas]
        X_val = self.data.loc[val_idx, feas]
        y_tr = self.data.loc[tr_idx, 'new_target']
        y_val = self.data.loc[val_idx, 'new_target']
        
        logger.info(f"Data split: train={len(X_tr)}, val={len(X_val)}")
        
        return X_tr, X_val, y_tr, y_val, tr_orgidx, val_orgidx
    
    def _create_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space for Optuna.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Dictionary of LightGBM parameters
        """
        params = self.base_params.copy()
        
        # Tree structure parameters
        params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
        params['num_leaves'] = 2 ** params['max_depth'] - 1
        params['min_child_samples'] = trial.suggest_int('min_child_samples', 10, 100)
        params['min_child_weight'] = trial.suggest_float('min_child_weight', 1e-3, 10, log=True)
        
        # Regularization parameters
        params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True)
        params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
        
        # Sampling parameters
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        
        # Learning parameters
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        
        # Class imbalance handling
        params['balanced_badrate'] = trial.suggest_float('balanced_badrate', 0.3, 2.0)
        
        # Early stopping
        params['stopping_rounds'] = self.early_stopping_rounds
        
        return params
    
    def _train_single_institution(
        self,
        org: Any,
        param: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train model for one institution using LOIO strategy.
        
        Args:
            org: Institution identifier
            param: LightGBM parameters
        
        Returns:
            Dictionary with training, validation, and OOS metrics
        """
        balanced_badrate = param.get('balanced_badrate', 1.0)
        
        # Prepare indices
        tr_idxs = set(self.X_tr.index)
        val_idxs = set(self.X_val.index)
        tr_idx = set(self.tr_orgidx.get(org, []))
        val_idx = set(self.val_orgidx.get(org, []))
        
        # Leave-one-out: exclude current institution
        X_tr_ = self.X_tr.loc[list(tr_idxs - tr_idx)]
        y_tr_ = self.y_tr.loc[list(tr_idxs - tr_idx)]
        
        X_val_ = self.X_val.loc[list(val_idxs - val_idx)]
        y_val_ = self.y_val.loc[list(val_idxs - val_idx)]
        
        # OOS: current institution's data
        X_oos = pd.concat([
            self.X_tr.loc[list(tr_idx)],
            self.X_val.loc[list(val_idx)]
        ])
        y_oos = pd.concat([
            self.y_tr.loc[list(tr_idx)],
            self.y_val.loc[list(val_idx)]
        ])
        
        # Handle edge case where OOS equals validation
        if len(y_val_) == len(y_oos) and len(y_oos) > 0:
            X_oos = X_oos.iloc[1:]
            y_oos = y_oos.iloc[1:]
        
        # Create sample weights (balanced bad rate)
        w_tr = pd.Series(
            np.where(y_tr_ == 1, balanced_badrate, 1.0),
            index=y_tr_.index
        )
        w_val = pd.Series(
            np.where(y_val_ == 1, balanced_badrate, 1.0),
            index=y_val_.index
        )
        w_oos = pd.Series(np.ones(len(y_oos)), index=y_oos.index)
        
        # Create LightGBM datasets
        train_set = lgb.Dataset(X_tr_, label=y_tr_, weight=w_tr)
        val_set = lgb.Dataset(X_val_, label=y_val_, reference=train_set)
        oos_set = lgb.Dataset(X_oos, label=y_oos, reference=train_set)
        
        # Prepare callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=param.get('stopping_rounds', 50))
        ]
        
        eval_results = {}
        if self.record_train_process:
            callbacks.append(lgb.record_evaluation(eval_results))
        
        # Train model
        model = lgb.train(
            param,
            train_set=train_set,
            valid_sets=[train_set, val_set, oos_set],
            valid_names=['train', 'val', 'oos'],
            callbacks=callbacks
        )
        
        # Calculate metrics
        pred_tr = model.predict(X_tr_)
        pred_val = model.predict(X_val_)
        pred_oos = model.predict(X_oos)
        
        train_metrics = _calculate_weighted_metrics(y_tr_, pred_tr, w_tr)
        val_metrics = _calculate_weighted_metrics(y_val_, pred_val, w_val)
        oos_metrics = _calculate_weighted_metrics(y_oos, pred_oos, w_oos)
        
        result = {
            'org': org,
            'train': train_metrics,
            'val': val_metrics,
            'oos': oos_metrics,
            'best_iteration': model.best_iteration
        }
        
        if self.record_train_process and len(eval_results) > 0:
            result['eval_history'] = pd.DataFrame(eval_results)
        
        return result
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial
        
        Returns:
            Objective value (negative mean validation KS if stable, else inf)
        """
        start_time = time.time()
        
        # Sample hyperparameters
        param = self._create_search_space(trial)
        
        # Remove custom params not for LightGBM
        param_clean = {k: v for k, v in param.items() 
                      if k not in ['balanced_badrate', 'stopping_rounds']}
        
        # Parallel training across institutions
        results = []
        with Pool(self.n_jobs) as pool:
            tasks = [(org, param) for org in self.tr_orgidx.keys()]
            results = pool.starmap(self._train_single_institution, tasks)
        
        # Aggregate metrics
        val_ks_list = [r['val']['ks'] for r in results]
        val_ks_w_list = [r['val']['ks_w'] for r in results]
        train_ks_list = [r['train']['ks'] for r in results]
        train_ks_w_list = [r['train']['ks_w'] for r in results]
        train_auc_list = [r['train']['auc'] for r in results]
        val_auc_list = [r['val']['auc'] for r in results]
        
        # Calculate mean metrics
        mean_val_ks = np.mean(val_ks_list)
        mean_val_ks_w = np.mean(val_ks_w_list)
        mean_oos_ks = np.mean([r['oos']['ks'] for r in results])
        
        # Stability check: train vs val shouldn't differ too much
        train_val_ks_diff = np.abs(np.array(train_ks_list) - np.array(val_ks_list))
        train_val_auc_diff = np.abs(np.array(train_auc_list) - np.array(val_auc_list))
        
        # Check stability conditions
        is_stable_auc = np.allclose(train_auc_list, val_auc_list, rtol=self.auc_threshold)
        is_stable_ks = np.allclose(train_ks_list, val_ks_list, rtol=self.ks_threshold)
        
        # Check OOS stability (train vs OOS)
        train_oos_ks_diff = np.abs(np.array(train_ks_list) - np.array([r['oos']['ks'] for r in results]))
        is_stable_oos = np.mean(train_oos_ks_diff) < 0.1  # Allow 10% difference
        
        elapsed_time = time.time() - start_time
        
        if is_stable_auc and is_stable_ks and is_stable_oos:
            loss = -mean_val_ks  # Negative because Optuna maximizes
            trial.set_user_attr('mean_val_ks', mean_val_ks)
            trial.set_user_attr('mean_oos_ks', mean_oos_ks)
            trial.set_user_attr('is_stable', True)
            logger.info(f"Trial {trial.number}: KS={mean_val_ks:.4f}, "
                       f"OOS_KS={mean_oos_ks:.4f}, Time={elapsed_time:.1f}s")
        else:
            loss = float('inf')
            trial.set_user_attr('is_stable', False)
            stability_issues = []
            if not is_stable_auc:
                stability_issues.append('AUC')
            if not is_stable_ks:
                stability_issues.append('KS')
            if not is_stable_oos:
                stability_issues.append('OOS')
            logger.warning(f"Trial {trial.number}: Unstable ({', '.join(stability_issues)}), "
                          f"Time={elapsed_time:.1f}s")
        
        return loss
    
    def optimize(
        self,
        timeout: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
        show_progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            timeout: Maximum time in seconds (optional)
            callbacks: List of Optuna callbacks (e.g., pruning)
            show_progress_bar: Whether to show progress bar
        
        Returns:
            Best parameters and study results
        """
        logger.info(f"Starting optimization: {self.max_iterations} trials, "
                   f"direction={self.direction}")
        
        begin_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"Start time: {begin_time}")
        
        self.study.optimize(
            self._objective,
            n_trials=self.max_iterations,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=show_progress_bar
        )
        
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"End time: {end_time}")
        
        # Get best parameters
        best_params = self.study.best_params.copy()
        best_params['balanced_badrate'] = best_params.get('balanced_badrate', 1.0)
        best_params['num_leaves'] = 2 ** best_params['max_depth'] - 1
        
        # Merge with base params
        final_params = {**self.base_params, **best_params}
        final_params = {k: v for k, v in final_params.items() 
                       if k not in ['balanced_badrate', 'stopping_rounds']}
        
        logger.info(f"Best trial: {self.study.best_trial.number}, "
                   f"Value={self.study.best_value:.4f}")
        
        return {
            'best_params': final_params,
            'best_value': self.study.best_value,
            'best_trial': self.study.best_trial,
            'study': self.study,
            'n_stable_trials': len([t for t in self.study.trials 
                                   if t.state == optuna.trial.TrialState.COMPLETE
                                   and t.user_attrs.get('is_stable', False)])
        }
    
    def get_trials_dataframe(self) -> pd.DataFrame:
        """
        Get trials results as DataFrame.
        
        Returns:
            DataFrame with all trial results
        """
        df = self.study.trials_dataframe()
        return df
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history.
        
        Args:
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
            fig = optuna.visualization.plot_optimization_history(self.study)
            if save_path:
                fig.write_image(save_path)
            return fig
        except Exception as e:
            logger.warning(f"Could not plot optimization history: {e}")
            return None
    
    def plot_param_importances(self, save_path: Optional[str] = None):
        """
        Plot parameter importance.
        
        Args:
            save_path: Path to save figure (optional)
        """
        try:
            fig = optuna.visualization.plot_param_importances(self.study)
            if save_path:
                fig.write_image(save_path)
            return fig
        except Exception as e:
            logger.warning(f"Could not plot param importances: {e}")
            return None


def create_default_search_space() -> Dict[str, Any]:
    """
    Create default LightGBM search space configuration.
    
    Returns:
        Dictionary with base parameters and search space config
    """
    return {
        # Base parameters (fixed)
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        
        # Search space (will be tuned)
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'num_leaves': {'type': 'int', 'low': 7, 'high': 255},
        'min_child_samples': {'type': 'int', 'low': 10, 'high': 100},
        'min_child_weight': {'type': 'float', 'low': 1e-3, 'high': 10, 'log': True},
        'reg_alpha': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
        'reg_lambda': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
        'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'balanced_badrate': {'type': 'float', 'low': 0.3, 'high': 2.0}
    }


# Convenience function for quick optimization
def quick_optimize_lgb(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'new_target',
    org_col: str = 'new_org',
    max_iterations: int = 50,
    n_jobs: int = 4,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick LightGBM optimization with minimal configuration.
    
    Args:
        data: Input DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        org_col: Organization/institution column name
        max_iterations: Number of Optuna trials
        n_jobs: Number of parallel jobs
        **kwargs: Additional parameters for OptunaLGB
    
    Returns:
        Optimization results dictionary
    """
    # Prepare data
    df = data[feature_cols + [target_col, org_col]].copy()
    
    # Base parameters
    base_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': kwargs.get('randn', 42),
        'n_estimators': 500,
        'force_col_wise': True
    }
    
    # Create optimizer
    optimizer = OptunaLGB(
        data=df,
        params=base_params,
        max_iterations=max_iterations,
        n_jobs=n_jobs,
        **kwargs
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    return results


if __name__ == '__main__':
    # Example usage
    logger.info("OptunaLGB module loaded. Use quick_optimize_lgb() for quick start.")