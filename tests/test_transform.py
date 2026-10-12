"""
tests/test_transform.py
=======================
اختبارات وحدة لمرحلة التنظيف src/data/transform.py.

بنختبر حاجتين:
  1) to_snake_case : دالة نقية (نص داخل -> نص خارج)، أسهل نوع اختبار.
  2) clean()       : بناخد داتا "خام" صغيرة بنفس أسماء أعمدة الـ CSV الأصلي،
                     ونتأكد إن اللي طلع هو اللي وعدنا بيه الكود: أسماء موحدة،
                     تكرار اتشال، فراغ اتملى، dtypes مضبوطة، وإن أي صف شاذ
                     بيتسجل في اللوج من غير ما يتمسح.

زي test_validate.py: مفيش قراءة ملفات، الداتا بتتبني بإيدينا.

التشغيل من جذر المشروع:
    python -m pytest tests/test_transform.py -v
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.data.transform import clean, to_snake_case
from src.data.validate import EXPECTED_COLUMNS, validate


# ── 1) to_snake_case ────────────────────────────────────────────────────
# كل صف = (الاسم الخام في الـ CSV، الاسم المتوقع بعد التحويل).
# بيغطي الحالات التلاتة: CamelCase، كلمات بمسافة، واسم جاهز أصلاً.
@pytest.mark.parametrize(
    "raw_name, expected",
    [
        ("CustomerId", "customer_id"),            # CamelCase
        ("CreditScore", "credit_score"),
        ("NumOfProducts", "num_of_products"),
        ("HasCrCard", "has_cr_card"),
        ("Satisfaction Score", "satisfaction_score"),   # مسافة
        ("Card Type", "card_type"),
        ("Geography", "geography"),               # كلمة واحدة
        ("already_snake_case", "already_snake_case"),   # idempotent: مفيش ضرر لو اتحول مرتين
        ("Double  Space", "double_space"),        # مسافتين ورا بعض -> _ واحدة
    ],
)
def test_to_snake_case(raw_name, expected):
    assert to_snake_case(raw_name) == expected


# ── Fixture: داتا خام صغيرة بأسماء الـ CSV الأصلية ───────────────────────
@pytest.fixture
def raw_df() -> pd.DataFrame:
    """5 عملاء، نظاف تمامًا، بأسماء الأعمدة زي ما هي في ملف Kaggle."""
    return pd.DataFrame(
        {
            "RowNumber": [1, 2, 3, 4, 5],
            "CustomerId": [101, 102, 103, 104, 105],
            "Surname": ["A", "B", "C", "D", "E"],
            "CreditScore": [600, 650, 700, 750, 800],
            "Geography": ["France", "Germany", "Spain", "France", "France"],
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
            "Age": [30, 40, 50, 35, 45],
            "Tenure": [1, 2, 3, 4, 5],
            "Balance": [0.0, 1000.0, 2000.0, 3000.0, 4000.0],
            "NumOfProducts": [1, 2, 1, 2, 1],
            "HasCrCard": [1, 0, 1, 1, 0],
            "IsActiveMember": [1, 1, 0, 1, 0],
            "EstimatedSalary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            "Exited": [0, 1, 0, 0, 1],
            "Complain": [0, 0, 0, 0, 1],
            "Satisfaction Score": [3, 4, 2, 5, 1],
            "Card Type": ["GOLD", "SILVER", "DIAMOND", "PLATINUM", "GOLD"],
            "Point Earned": [300, 400, 500, 600, 700],
        }
    )


# ── 2) clean(): الشكل العام ─────────────────────────────────────────────
def test_clean_renames_columns_and_drops_row_number(raw_df):
    out = clean(raw_df)
    assert "row_number" not in out.columns
    assert set(out.columns) == EXPECTED_COLUMNS


def test_clean_output_passes_validation(raw_df):
    """
    اختبار "تكامل" صغير: ناتج clean() لازم يعدّي بوابة validate().
    لو حد غيّر transform بطريقة تكسر توقعات validate (مثلاً غيّر اسم
    عمود)، الاختبار ده يمسكها قبل ما الـ pipeline يقع.
    """
    assert validate(clean(raw_df)) is None


def test_clean_does_not_modify_input(raw_df):
    """df.copy() في أول clean() لها غرض: الـ DataFrame الأصلي يفضل سليم."""
    snapshot = raw_df.copy(deep=True)
    clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, snapshot)


# ── 3) clean(): التكرار والفراغ ──────────────────────────────────────────
def test_clean_drops_duplicate_customer_ids_keeping_first(raw_df):
    dup = raw_df.copy()
    dup.loc[4, "CustomerId"] = 101          # الصف الأخير يكرر id الصف الأول
    out = clean(dup)
    assert len(out) == 4
    assert out["customer_id"].is_unique
    # keep="first": الصف الأصلي (Surname A) هو اللي بيفضل
    assert out.loc[out["customer_id"] == 101, "surname"].iloc[0] == "A"


def test_clean_fills_numeric_nulls_with_median(raw_df):
    nulls = raw_df.copy()
    nulls.loc[0, "Balance"] = np.nan
    # قيمة شاذة (10000) عن قصد: الـ median = 2500 لكن الـ mean = 4000.
    # لو الكود غلط واستخدم mean، الاختبار يفشل. (قيم متساوية الفرق كانت
    # هتخلي الاتنين يطلعوا نفس الرقم، والاختبار مكانش هيكشف الغلطة.)
    nulls.loc[1:4, "Balance"] = [1000.0, 2000.0, 3000.0, 10000.0]
    out = clean(nulls)
    assert out.loc[out["customer_id"] == 101, "balance"].iloc[0] == 2500.0
    assert out["balance"].isnull().sum() == 0


def test_clean_fills_categorical_nulls_with_mode(raw_df):
    nulls = raw_df.copy()
    nulls.loc[1, "Geography"] = None        # الأكتر تكرارًا بين الباقي: France (3 مرات)
    out = clean(nulls)
    assert out.loc[out["customer_id"] == 102, "geography"].iloc[0] == "France"


# ── 4) clean(): قرار "سجّل ولا تمسح" ─────────────────────────────────────
def test_out_of_range_rows_are_logged_not_dropped(raw_df, caplog):
    odd = raw_df.copy()
    odd.loc[0, "Age"] = 5                   # برّه النطاق (18-100)
    # caplog = أداة pytest بتلتقط رسائل الـ logging عشان نفحصها
    with caplog.at_level(logging.WARNING, logger="src.data.transform"):
        out = clean(odd)
    assert len(out) == len(raw_df)          # الصف لسه موجود
    assert "out of range for age" in caplog.text   # والتحذير اتسجل


# ── 5) clean(): الـ dtypes ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "column, expected_dtype",
    [
        ("customer_id", "int64"),
        ("age", "int8"),
        ("credit_score", "int16"),
        ("point_earned", "int16"),
        ("exited", "int8"),
        ("has_cr_card", "int8"),
        ("balance", "float64"),
        ("geography", "category"),
        ("card_type", "category"),
    ],
)
def test_clean_sets_expected_dtypes(raw_df, column, expected_dtype):
    out = clean(raw_df)
    assert str(out[column].dtype) == expected_dtype


# ── 6) clean(): تنظيف المسافات الزيادة ───────────────────────────────────
def test_clean_strips_whitespace_in_text_columns(raw_df):
    messy = raw_df.copy()
    messy.loc[0, "Geography"] = "  France "
    out = clean(messy)
    # بعد الـ strip، " France " و "France" نفس القيمة => 3 فئات بس مش 4
    assert set(out["geography"].cat.categories) == {"France", "Germany", "Spain"}