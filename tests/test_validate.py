"""
tests/test_validate.py
======================
اختبارات وحدة (unit tests) لبوابة الجودة src/data/validate.py.

الفكرة: validate() مفروض توقف الـ pipeline (DataValidationError) لما الداتا
فيها مشكلة، وتعدّي بهدوء لما الداتا سليمة. هنا بنتأكد من الاتنين:
  1) داتا سليمة      -> مفيش exception
  2) كل نوع مشكلة    -> exception بالرسالة المناسبة

الاختبارات دي مبتقراش أي ملف (لا CSV ولا parquet): بنبني DataFrame صغير
بإيدينا. ده بيخليها سريعة (أجزاء من الثانية) وتشتغل على أي جهاز وفي CI
من غير ما نحتاج الداتا الأصلية.

التشغيل من جذر المشروع:
    python -m pytest tests/test_validate.py -v
"""

import pandas as pd
import pytest

from src.data.validate import (
    DataValidationError,
    EXPECTED_COLUMNS,
    validate,
)


# ── Fixture: داتا سليمة صغيرة ───────────────────────────────────────────
# @pytest.fixture = دالة بتجهّز "مادة" الاختبار. كل اختبار بيطلبها باسمها
# كـ argument، وpytest بيبنيها من جديد لكل اختبار — فأي اختبار يخرّب
# الداتا (يحذف عمود مثلاً) مبيأثرش على اللي بعده.
@pytest.fixture
def valid_df() -> pd.DataFrame:
    """4 عملاء، واحد منهم churned: كفاية عشان كل الفحوصات تعدّي."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "surname": ["A", "B", "C", "D"],
            "credit_score": [600, 650, 700, 750],
            "geography": ["France", "Germany", "Spain", "France"],
            "gender": ["Male", "Female", "Male", "Female"],
            "age": [30, 40, 50, 35],
            "tenure": [1, 2, 3, 4],
            "balance": [0.0, 1000.0, 2000.0, 3000.0],
            "num_of_products": [1, 2, 1, 2],
            "has_cr_card": [1, 0, 1, 1],
            "is_active_member": [1, 1, 0, 1],
            "estimated_salary": [50000.0, 60000.0, 70000.0, 80000.0],
            # لازم 0 و 1 مع بعض، وإلا فحص نسبة الـ churn هيفشل (انظر آخر اختبار)
            "exited": [0, 1, 0, 0],
            "complain": [0, 0, 0, 0],
            "satisfaction_score": [3, 4, 2, 5],
            "card_type": ["GOLD", "SILVER", "DIAMOND", "PLATINUM"],
            "point_earned": [300, 400, 500, 600],
        }
    )


def test_fixture_has_all_expected_columns(valid_df):
    """
    حارس للـ fixture نفسها: لو حد أضاف عمود لـ EXPECTED_COLUMNS ونسي يحدّث
    الـ fixture، الاختبار ده هو اللي يقولك "المشكلة في الاختبار مش في الكود".
    """
    assert set(valid_df.columns) == EXPECTED_COLUMNS


# ── 1) الحالة السليمة ───────────────────────────────────────────────────
def test_valid_data_passes(valid_df):
    # validate() بترجّع None لو كل حاجة تمام. لو رمت exception،
    # pytest بيعتبر الاختبار فشل تلقائيًا، فمحتاجين assert هنا.
    assert validate(valid_df) is None


# ── 2) كل نوع مشكلة لازم يوقف الـ pipeline ───────────────────────────────
def test_missing_column_is_rejected(valid_df):
    broken = valid_df.drop(columns=["age"])
    # pytest.raises = "أتوقع exception من النوع ده". match= بيفحص إن الرسالة
    # فيها النص ده (regex)، فنتأكد إنه فشل للسبب الصح مش لسبب تاني.
    with pytest.raises(DataValidationError, match="Missing expected columns"):
        validate(broken)


def test_null_value_is_rejected(valid_df):
    broken = valid_df.copy()
    broken.loc[0, "balance"] = None
    with pytest.raises(DataValidationError, match="Unexpected nulls"):
        validate(broken)


def test_duplicate_customer_id_is_rejected(valid_df):
    broken = valid_df.copy()
    broken.loc[1, "customer_id"] = 1   # نفس id الصف الأول
    with pytest.raises(DataValidationError, match="duplicate customer_id"):
        validate(broken)


# parametrize = نفس الاختبار يتكرر على أكتر من مدخل (3 اختبارات من كود واحد).
# كل عمود categorical لازم يرفض قيمة غريبة عليه.
@pytest.mark.parametrize(
    "column, bad_value",
    [
        ("geography", "Italy"),
        ("gender", "Other"),
        ("card_type", "BRONZE"),
    ],
)
def test_unknown_category_is_rejected(valid_df, column, bad_value):
    broken = valid_df.copy()
    broken.loc[0, column] = bad_value
    with pytest.raises(DataValidationError, match=f"Unexpected values in '{column}'"):
        validate(broken)


# نسبة churn = 0% أو 100% معناها غالبًا خطأ في المصدر، مش داتا "نضيفة أكتر".
@pytest.mark.parametrize("exited_values", [[0, 0, 0, 0], [1, 1, 1, 1]])
def test_implausible_churn_rate_is_rejected(valid_df, exited_values):
    broken = valid_df.copy()
    broken["exited"] = exited_values
    with pytest.raises(DataValidationError, match="outside plausible bounds"):
        validate(broken)


# ── 3) ترتيب الفحوصات ──────────────────────────────────────────────────
def test_stops_at_first_failure(valid_df):
    """
    validate() مصممة توقف عند أول مشكلة. هنا الداتا فيها مشكلتين:
    عمود ناقص + قيمة فارغة. المتوقع إن رسالة العمود الناقص هي اللي تظهر
    (لأن validate_schema بتتنفذ أولاً)، مش رسالة الـ nulls.
    """
    broken = valid_df.drop(columns=["age"])
    broken.loc[0, "balance"] = None
    with pytest.raises(DataValidationError, match="Missing expected columns"):
        validate(broken)