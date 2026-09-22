"""
Optuna hyperparameter search for XGBoost, optimizing balanced accuracy
directly (not a proxy metric such as log-loss) on a single stratified
train/val split for speed. Full cross-validation with the resulting best
parameters is performed separately in train.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from preprocess import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    TARGET,
    encode_categorical,
    encode_target,
    load_and_clean,
)

RANDOM_STATE = 42

# Early-stopping patience of 200 rounds is required: balanced accuracy near
# convergence improves slowly with small noise, and a smaller patience (e.g.
# 50) was found to trigger false early stops at local noise peaks.
STOPPING_ROUNDS = 200


def feval_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Custom eval metric for XGBoost's sklearn API (must return a single
    float; the metric's internal name is derived from this function's
    __name__, i.e. 'feval_balanced_accuracy')."""
    pred_labels = np.argmax(y_pred, axis=1)
    return balanced_accuracy_score(y_true, pred_labels)


def build_objective(X_train, y_train, X_val, y_val, sample_weight, device: str):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": feval_balanced_accuracy,
            "device": device,
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

        model = xgb.XGBClassifier(
            **params,
            callbacks=[xgb.callback.EarlyStopping(
                rounds=STOPPING_ROUNDS, metric_name="feval_balanced_accuracy",
                maximize=True, save_best=True,
            )],
        )
        model.fit(
            X_train, y_train, sample_weight=sample_weight,
            eval_set=[(X_val, y_val)], verbose=False,
        )

        val_pred = np.argmax(model.predict_proba(X_val), axis=1)
        return balanced_accuracy_score(y_val, val_pred)

    return objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=7200, help="Seconds")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="best_params.json")
    args = parser.parse_args()

    train_df, _ = load_and_clean(args.train, args.train)  # test path unused here
    y_encoded, label_to_code = encode_target(train_df[TARGET])

    train_idx, val_idx = train_test_split(
        train_df.index, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
    )
    train_split, val_split = train_df.loc[train_idx].reset_index(drop=True), train_df.loc[val_idx].reset_index(drop=True)
    y_train, y_val = y_encoded.loc[train_idx].reset_index(drop=True), y_encoded.loc[val_idx].reset_index(drop=True)

    encoded = encode_categorical(fit_df=train_split, transform_dfs={"train": train_split, "val": val_split})
    X_train, X_val = encoded["train"][FEATURE_COLS], encoded["val"][FEATURE_COLS]

    class_labels = np.array(sorted(label_to_code.values()))
    weights = compute_class_weight(class_weight="balanced", classes=class_labels, y=y_train)
    sample_weight = y_train.map(dict(zip(class_labels, weights))).values

    objective = build_objective(X_train, y_train, X_val, y_val, sample_weight, args.device)

    study = optuna.create_study(direction="maximize", study_name="xgb_balanced_accuracy")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print(f"Best balanced accuracy: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    with open(args.output, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved best params to {args.output}")


if __name__ == "__main__":
    main()