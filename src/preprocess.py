"""
Preprocessing pipeline for the Student Health Risk classification task.

Design decisions (see notebooks/eda.ipynb for supporting analysis):
- No imputation: XGBoost handles missing values natively via learned split
  direction. IQR outlier rate is low (<2%) and values are physiologically
  plausible, so no outlier removal is performed either.
- Categorical encoding: `sleep_quality` uses ordinal encoding because its
  monotonic relationship with the target was verified via Spearman
  correlation in EDA. The remaining categorical columns show no verified
  monotonic order, so they use plain label encoding.
"""

from __future__ import annotations

import pandas as pd

TARGET = "health_condition"
ID_COL = "id"

NUMERIC_COLS = [
    "sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
    "step_count", "exercise_duration", "water_intake",
]
CATEGORICAL_COLS = [
    "diet_type", "stress_level", "sleep_quality",
    "physical_activity_level", "smoking_alcohol", "gender",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

HYPOTHESIZED_ORDERS: dict[str, list[str]] = {
    "stress_level": ["low", "medium", "high"],
    "sleep_quality": ["poor", "average", "good"],
    "physical_activity_level": ["sedentary", "moderate", "active"],
}

# Verified via Spearman monotonicity check against the target in EDA.
# Only sleep_quality showed a fully monotonic relationship across all
# 3 target classes.
ORDINAL_VERIFIED: dict[str, bool] = {
    "stress_level": False,
    "sleep_quality": True,
    "physical_activity_level": False,
}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """No-op cleaning step, kept explicit to document the decision to
    preserve NaN and outliers for the tree-based model."""
    return df.copy()


def encode_categorical(
    fit_df: pd.DataFrame,
    transform_dfs: dict[str, pd.DataFrame],
    cat_cols: list[str] = CATEGORICAL_COLS,
    hypothesized_orders: dict[str, list[str]] = HYPOTHESIZED_ORDERS,
    ordinal_verified: dict[str, bool] = ORDINAL_VERIFIED,
) -> dict[str, pd.DataFrame]:
    """
    Encode categorical columns for XGBoost.

    Ordinal encoding is applied to columns with a verified monotonic
    relationship with the target; all other categorical columns use label
    encoding (integer codes with no implied order). NaN is preserved as
    float NaN, which XGBoost routes via its learned default split direction.

    Args:
        fit_df: DataFrame used to determine the encoding (e.g. a fold's
            training split), preventing leakage from validation/test data.
        transform_dfs: named DataFrames to transform using fit_df's mapping,
            e.g. {"train": ..., "val": ..., "test": ...}.

    Returns:
        Dict of encoded DataFrames matching the keys of transform_dfs.
    """
    encoded = {name: df.copy() for name, df in transform_dfs.items()}

    for col in cat_cols:
        if ordinal_verified.get(col, False):
            order = hypothesized_orders[col]
            rank_map = {cat: i for i, cat in enumerate(order)}
            for name, df in transform_dfs.items():
                encoded[name][col] = df[col].map(rank_map)
        else:
            categories = fit_df[col].dropna().unique().tolist()
            label_map = {cat: i for i, cat in enumerate(categories)}
            for name, df in transform_dfs.items():
                encoded[name][col] = df[col].map(label_map)

    for name in encoded:
        encoded[name][cat_cols] = encoded[name][cat_cols].astype(float)

    return encoded


def encode_target(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Map string target labels to integer class codes (alphabetical order)."""
    classes = sorted(series.unique().tolist())
    label_to_code = {label: i for i, label in enumerate(classes)}
    return series.map(label_to_code), label_to_code


def load_and_clean(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = clean_data(pd.read_csv(train_path))
    test_df = clean_data(pd.read_csv(test_path))
    return train_df, test_df