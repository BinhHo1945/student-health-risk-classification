"""
Train the final XGBoost model with 5-fold Stratified K-Fold cross-validation
using tuned hyperparameters, and generate the competition submission file.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from preprocess import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    ID_COL,
    TARGET,
    encode_categorical,
    encode_target,
    load_and_clean,
)

RANDOM_STATE = 42
N_SPLITS = 5
STOPPING_ROUNDS = 200


def feval_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pred_labels = np.argmax(y_pred, axis=1)
    return balanced_accuracy_score(y_true, pred_labels)


def train_kfold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_params: dict,
    device: str = "cuda",
) -> tuple[pd.Series, float]:
    """Run 5-fold Stratified K-Fold training, return test predictions
    (averaged across folds) and the OOF balanced accuracy score."""
    y_labels = train_df[TARGET]
    y_encoded, label_to_code = encode_target(y_labels)
    code_to_label = {v: k for k, v in label_to_code.items()}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred_codes = np.zeros(len(train_df), dtype=int)
    test_proba_sum = np.zeros((len(test_df), len(label_to_code)))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, y_encoded), start=1):
        train_fold = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold = train_df.iloc[val_idx].reset_index(drop=True)
        y_train_fold = y_encoded.iloc[train_idx].reset_index(drop=True)
        y_val_fold = y_encoded.iloc[val_idx].reset_index(drop=True)

        encoded = encode_categorical(
            fit_df=train_fold,
            transform_dfs={"train": train_fold, "val": val_fold, "test": test_df},
        )
        X_train, X_val, X_test = (
            encoded["train"][FEATURE_COLS], encoded["val"][FEATURE_COLS], encoded["test"][FEATURE_COLS]
        )

        class_labels = np.array(sorted(label_to_code.values()))
        weights = compute_class_weight(class_weight="balanced", classes=class_labels, y=y_train_fold)
        sample_weight = y_train_fold.map(dict(zip(class_labels, weights))).values

        model = xgb.XGBClassifier(
            objective="multi:softprob", num_class=len(label_to_code),
            eval_metric=feval_balanced_accuracy, device=device, tree_method="hist",
            random_state=RANDOM_STATE, n_estimators=3000,
            callbacks=[xgb.callback.EarlyStopping(
                rounds=STOPPING_ROUNDS, metric_name="feval_balanced_accuracy",
                maximize=True, save_best=True,
            )],
            **best_params,
        )
        model.fit(
            X_train, y_train_fold, sample_weight=sample_weight,
            eval_set=[(X_val, y_val_fold)], verbose=False,
        )

        val_proba = model.predict_proba(X_val)
        oof_pred_codes[val_idx] = np.argmax(val_proba, axis=1)
        test_proba_sum += model.predict_proba(X_test)

        fold_score = balanced_accuracy_score(y_val_fold, np.argmax(val_proba, axis=1))
        print(f"Fold {fold_idx}/{N_SPLITS} — balanced accuracy: {fold_score:.4f} (best_iteration: {model.best_iteration})")

    oof_pred_labels = pd.Series(oof_pred_codes).map(code_to_label)
    oof_score = balanced_accuracy_score(y_labels, oof_pred_labels)

    print(f"\nOOF Balanced Accuracy: {oof_score:.4f}")
    print(classification_report(y_labels, oof_pred_labels))

    test_proba_avg = test_proba_sum / N_SPLITS
    test_pred_labels = pd.Series([code_to_label[i] for i in np.argmax(test_proba_avg, axis=1)])

    return test_pred_labels, oof_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--params", required=True, help="Path to best_params.json from tune.py")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    train_df, test_df = load_and_clean(args.train, args.test)
    with open(args.params) as f:
        best_params = json.load(f)

    test_pred_labels, oof_score = train_kfold(train_df, test_df, best_params, device=args.device)

    submission_df = pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET: test_pred_labels.values})
    submission_df.to_csv(args.output, index=False)
    print(f"\nSaved submission to {args.output} (OOF balanced accuracy: {oof_score:.4f})")


if __name__ == "__main__":
    main()