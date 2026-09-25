# src/model.py

import os
import joblib
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


class TabularMatcher:
    """
    Gradient-boosted tree matcher for entity-resolution pair features.

    The model receives engineered pairwise features such as:
      - name similarities
      - address similarities
      - token overlaps
      - digit similarities
      - exact/partial indicators

    This is generally a better inductive bias than applying Conv1D
    across an arbitrary feature vector.
    """

    def __init__(
        self,
        max_iter=350,
        learning_rate=0.06,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=30,
        )

        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int8)

        if len(X) == 0:
            raise ValueError("Cannot train matcher with zero samples.")

        if len(np.unique(y)) < 2:
            raise ValueError("Training data must contain both positive and negative examples.")

        self.feature_names = feature_names

        print(
            f"Training Gradient Boosting matcher on "
            f"{len(X):,} pairs and {X.shape[1]} features..."
        )

        self.model.fit(X, y)

        train_probs = self.model.predict_proba(X)[:, 1]

        try:
            auc = roc_auc_score(y, train_probs)
            print(f"Training ROC-AUC: {auc:.6f}")
        except Exception:
            pass

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.predict_proba(X)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        joblib.dump(
            {
                "model_type": "HistGradientBoostingMatcher",
                "model": self.model,
                "feature_names": self.feature_names,
                "config": {
                    "max_iter": self.max_iter,
                    "learning_rate": self.learning_rate,
                    "max_leaf_nodes": self.max_leaf_nodes,
                    "max_depth": self.max_depth,
                    "min_samples_leaf": self.min_samples_leaf,
                    "l2_regularization": self.l2_regularization,
                    "random_state": self.random_state,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)

        if isinstance(payload, dict) and "model" in payload:
            obj = cls(**payload.get("config", {}))
            obj.model = payload["model"]
            obj.feature_names = payload.get("feature_names")
            return obj

        # Backward compatibility if an older sklearn object was saved.
        obj = cls()
        obj.model = payload
        return obj


def train_matching_model(
    X_train,
    y_train,
    model_path="models/matcher.joblib",
    feature_names=None,
):
    model = TabularMatcher(
        max_iter=350,
        learning_rate=0.06,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        feature_names=feature_names,
    )

    model.save(model_path)

    print(f"Model saved to {model_path}")

    return model


def load_matching_model(model_path):
    print(f"Loading matcher from {model_path}")
    return TabularMatcher.load(model_path)


def fbeta_score_single(pred_set, true_set, beta=0.5):
    """
    F_beta for one S1 entity.

    Empty prediction + empty truth = 1.
    """

    pred_set = set(pred_set)
    true_set = set(true_set)

    if not pred_set and not true_set:
        return 1.0

    if not pred_set and true_set:
        return 0.0

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta2 = beta * beta

    return (
        (1.0 + beta2)
        * precision
        * recall
        / (beta2 * precision + recall)
    )


def compute_macro_f05(preds, gt_dict, s1_ids):
    """
    Macro F0.5 across ALL S1 entities.

    Important:
    S1 entities absent from gt_dict are treated as having
    an empty ground-truth match set.
    """

    scores = []

    for s1_id in s1_ids:
        predicted = set(preds.get(s1_id, []))
        truth = set(gt_dict.get(s1_id, set()))

        score = fbeta_score_single(
            predicted,
            truth,
            beta=0.5,
        )

        scores.append(score)

    if not scores:
        return 0.0

    return float(np.mean(scores))