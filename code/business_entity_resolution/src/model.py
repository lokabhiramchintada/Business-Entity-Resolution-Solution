import os
import joblib
import numpy as np
import lightgbm as lgb
from .features import FEATURE_NAMES

def train_matching_model(X_train, y_train, model_path=None):
    """Train a LightGBM gradient boosted tree classifier for entity resolution."""
    lgb_train = lgb.Dataset(np.array(X_train, dtype=np.float32), label=np.array(y_train, dtype=np.int32), feature_name=FEATURE_NAMES)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.85,
        'verbose': -1,
        'random_state': 42
    }
    
    model = lgb.train(params, lgb_train, num_boost_round=150)
    
    if model_path:
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        joblib.dump(model, model_path)
        
    return model

def load_matching_model(model_path):
    """Load pre-trained LightGBM model."""
    return joblib.load(model_path)

def compute_macro_f05(pred_dict, gt_dict, all_s1_ids):
    """
    Compute official challenge Macro F_0.5 score:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons score 1.0 if empty list predicted, else 0.0.
    """
    f05_scores = []
    for s1 in all_s1_ids:
        true_set = gt_dict.get(s1, set())
        pred_set = set(pred_dict.get(s1, []))
        
        if len(true_set) == 0:
            f05_scores.append(1.0 if len(pred_set) == 0 else 0.0)
            continue
            
        if len(pred_set) == 0:
            f05_scores.append(0.0)
            continue
            
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        if tp == 0:
            f05_scores.append(0.0)
            continue
            
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        denom = 0.25 * prec + rec
        score = (1.25 * prec * rec) / denom if denom > 0 else 0.0
        f05_scores.append(score)
        
    return float(np.mean(f05_scores))
