"""
tests/test_serving_consistency.py
=================================
اختبار "training-serving skew": هل الـ API بتحسب نفس الاحتمال اللي
الموديل بيحسبه وقت التدريب لنفس العميل؟

الفكرة:
-------
عندنا طريقتين لتجهيز الـ features:
  1) وقت التدريب: build_features(df) على الداتا كلها (دفعة واحدة).
  2) وقت التنبؤ: prepare_single_customer(dict) لعميل واحد (زي ما الـ API بتعمل).
لو الطريقتين طلّعوا نفس الأعمدة بنفس القيم، الاحتمال لازم يتطابق.
لو اتخالفوا (عمود اتسمّى غلط، ترتيب اتغيّر، قيمة اتملت صفر بالغلط)،
الموديل هيشوف بيانات مختلفة في الإنتاج، ومفيش خطأ ظاهر بيقولك.

التشغيل (من جذر المشروع):
    pip install pytest
    python -m pytest tests/test_serving_consistency.py -v

الاختبار بيتخطّى (skip) لو ملفات الداتا أو الموديل مش موجودة، بدل ما يفشل.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.paths import PROJECT_ROOT, load_config
from src.features.build_features import (
    get_feature_columns_path,
    get_features_output_path,
    prepare_single_customer,
)

# الحقول اللي الـ API بتستقبلها (نفس CustomerInput في schemas.py)
API_FIELDS = [
    "credit_score", "geography", "gender", "age", "tenure", "balance",
    "num_of_products", "has_cr_card", "is_active_member", "estimated_salary",
    "satisfaction_score", "card_type", "point_earned",
]
CATEGORICAL = ["geography", "gender", "card_type"]

# الموديلات اللي نختبرها، لو الملف موجود
MODEL_FILES = ["churn_model.joblib", "churn_model_xgboost.joblib"]

N_SAMPLES = 200
SEED = 0


def _clean_parquet_path():
    config = load_config()
    return PROJECT_ROOT / config["data"]["processed_dir"] / config["data"]["processed_filename"]


@pytest.fixture(scope="module")
def data():
    """الداتا الخام النظيفة + مصفوفة الـ features اللي اتدرّب عليها الموديل."""
    clean_path = _clean_parquet_path()
    features_path = get_features_output_path()
    columns_path = get_feature_columns_path()
    for p in (clean_path, features_path, columns_path):
        if not p.exists():
            pytest.skip(f"ملف ناقص: {p.name}. شغّل pipeline وbuild_features الأول.")

    clean_df = pd.read_parquet(clean_path)
    features_df = pd.read_parquet(features_path)
    with open(columns_path, encoding="utf-8") as f:
        feature_columns = json.load(f)

    # build_features بيحافظ على ترتيب الصفوف، فالصف i في الاتنين هو نفس العميل
    assert len(clean_df) == len(features_df)
    return clean_df, features_df, feature_columns


def test_feature_columns_match_training_matrix(data):
    """قايمة الأعمدة المحفوظة لازم تكون موجودة بالظبط في مصفوفة التدريب."""
    _, features_df, feature_columns = data
    missing = [c for c in feature_columns if c not in features_df.columns]
    assert not missing, f"أعمدة في feature_columns.json مش في مصفوفة التدريب: {missing}"
    assert "exited" not in feature_columns
    assert "complain" not in feature_columns


@pytest.mark.parametrize("model_file", MODEL_FILES)
def test_api_probability_equals_training_probability(data, model_file):
    """
    لكل عميل في عينة عشوائية: الاحتمال من مسار الـ API (عميل واحد) لازم
    يساوي الاحتمال من مصفوفة التدريب لنفس العميل.
    """
    model_path = PROJECT_ROOT / "joblib" / model_file
    if not model_path.exists():
        pytest.skip(f"الموديل {model_file} مش موجود")
    model = joblib.load(model_path)

    clean_df, features_df, feature_columns = data
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(clean_df), size=N_SAMPLES, replace=False)

    # طريقة التدريب: نفس الأعمدة بنفس الترتيب، من غير exited وcomplain
    training_X = features_df[feature_columns].iloc[idx]
    expected = model.predict_proba(training_X)[:, 1]

    # طريقة الـ API: dict لكل عميل ← prepare_single_customer ← الموديل
    actual = []
    for i in idx:
        raw = clean_df.iloc[i][API_FIELDS].to_dict()
        for col in CATEGORICAL:
            raw[col] = str(raw[col])  # الأعمدة دي category dtype في الـ parquet
        row = prepare_single_customer(raw, feature_columns)
        actual.append(float(model.predict_proba(row)[0][1]))

    np.testing.assert_allclose(actual, expected, atol=1e-6)