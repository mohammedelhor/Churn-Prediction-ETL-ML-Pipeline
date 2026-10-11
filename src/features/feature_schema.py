
"""
Feature Contract
================
Central definition of the source schema and model feature policy
for the customer churn prediction project.

This module defines metadata only.
It does not clean data, encode categories, or train models.
"""

# Target to predict
TARGET_COLUMN = "exited"

# Columns identifying a customer, not predictive features
IDENTIFIER_COLUMNS = [
    "customer_id",
    "surname",
]

# Excluded because they may introduce target leakage or be unavailable
# at prediction time. Reassess if the business use case changes.

EXPERIMENTAL_FEATURES = ["complain"]

# Categorical columns requiring encoding
CATEGORICAL_FEATURES = [
    "geography",
    "gender",
    "card_type",
]

# Numeric predictors
NUMERIC_FEATURES = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "num_of_products",
    "estimated_salary",
    "satisfaction_score",
    "point_earned",
]

# Binary predictors represented as 0/1
BINARY_FEATURES = [
    "has_cr_card",
    "is_active_member",
]

# Features created during feature engineering
ENGINEERED_FEATURES = [
    "is_zero_balance",
    "balance_per_product",
]

# All columns expected in the cleaned source dataset.
# This includes target, identifiers, and excluded columns.
SOURCE_COLUMNS = [
    "customer_id",
    "surname",
    "credit_score",
    "geography",
    "gender",
    "age",
    "tenure",
    "balance",
    "num_of_products",
    "has_cr_card",
    "is_active_member",
    "estimated_salary",
    TARGET_COLUMN,
    "complain",
    "satisfaction_score",
    "card_type",
    "point_earned",
]

# All raw columns allowed to serve as model predictors.
# Target, identifiers, and leakage columns are deliberately excluded.
RAW_MODEL_FEATURES = (
    NUMERIC_FEATURES
    + BINARY_FEATURES
    + CATEGORICAL_FEATURES
)

# Category values currently accepted by the data contract.
# These should be updated intentionally when new valid categories appear.
ALLOWED_CATEGORIES = {
    "geography": {"France", "Germany", "Spain"},
    "gender": {"Male", "Female"},
    "card_type": {"DIAMOND", "GOLD", "SILVER", "PLATINUM"},
}

# Derived features plus the raw predictors form the pre-encoding feature set.
PRE_ENCODING_FEATURES = (
    RAW_MODEL_FEATURES + ENGINEERED_FEATURES
)
