# Student Health Risk Classification — Kaggle Playground Series S6E7

Multi-class classification predicting student health condition (`at-risk` / `fit` / `unhealthy`)
from behavioral and physiological features. Competition: [Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7).

## Result

- **Model**: XGBoost (GPU, Optuna-tuned)
- **CV strategy**: 5-fold Stratified K-Fold, early stopping on balanced accuracy
- **OOF Balanced Accuracy**: 0.9501
- **Private Leaderboard**: 0.95009
- **Public Leaderboard**: 0.94946

## Dataset

- 690,088 train rows / 295,753 test rows, 13 features (7 numeric, 6 categorical)
- Severe class imbalance: at-risk 85.9% / unhealthy 8.4% / fit 5.8%
- Missing values across most columns (1–12%), handled natively by the tree model (no imputation)

## Approach

1. **EDA** (`notebooks/eda.ipynb`): missing-data mechanism analysis, target-class correlation,
   ordinal-order verification for categorical features via Spearman monotonicity.
2. **Preprocessing** (`src/preprocess.py`): minimal cleaning (NaN preserved, no outlier removal —
   justified by EDA); categorical encoding — ordinal for `sleep_quality` (verified monotonic),
   label encoding for the rest.
3. **Tuning** (`src/tune.py`): Optuna Bayesian search (30 trials, single train/val split) optimizing
   balanced accuracy directly, not a proxy metric like log-loss.
4. **Training** (`src/train.py`): 5-fold Stratified K-Fold with `class_weight="balanced"`, GPU-accelerated,
   early stopping on balanced accuracy (200-round patience — required because balanced accuracy improves
   slowly with small noise near convergence).

## Key methodology notes

- Early stopping must track the competition metric (balanced accuracy) directly. Using a proxy metric
  (log-loss) caused early stopping to optimize in the wrong direction under class imbalance.
- A too-small `early_stopping_rounds` (50) caused false early stops on local noise; increased to 200
  after diagnosing the learning curve.
- Ensembling with LightGBM/CatBoost and feature engineering (categorical interactions, ratio features)
  were evaluated via OOF but did not improve over this single tuned XGBoost model — see project notes
  for details. This repo contains only the best-performing pipeline.

## How to run

```bash
pip install -r requirements.txt

# 1. Hyperparameter search (writes best_params.json)
python src/tune.py --train data/train.csv --n-trials 30 --timeout 7200

# 2. Train final model with 5-fold CV, generate submission.csv
python src/train.py --train data/train.csv --test data/test.csv --params best_params.json
```

## Requirements

See `requirements.txt`. GPU (CUDA) required for `device="cuda"` in XGBoost; set `device="cpu"` in
both scripts if no GPU is available (slower, results should remain equivalent).
