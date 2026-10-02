# import pandas as pd

# def validate_data(df: pd.DataFrame):
#     error = []

#     # 1. check if empty
#     if df.empty:
#        errors.append("Dataset is empty")

#     # 2. check missing values
#     missing = df.isnull().sum().sum()
#     print(f"Total missing values: {missing}")
#     print(df.isnull().sum())

#     # 3. check duplicates
#     duplicates = df.duplicates().sum()
#     print(f"Total duplicates: {duplicates}")

#     if duplicates > 0:
#         print(f"warning: {duplicates} duplicates rows found")

#     # 4. Basic schema check (example)
#     expected_columns = [
#         "Age"
#         "customer_id"
#         "surname"
#         "credit_score"
#         "geography"
#         "gender"
#         "tenure"
#         "balance"
#         "num_of_products"
#         "has_cr_card"
#         "is_active_member"
#         "estimated_salary"
#         "exited"
#         "complain"
#         "satisfaction_score"
#         "card_type"
#         "point_earned"
#     ]

#     for col in expected_columns : 
#         if col not in df.columns:
#            errors.append(f"Missing column: {col}")
#     if errors:
#         raise ValueError(f"Data validation failed: {errors}")
#     if not pd.api.types.is_numeric_dtype(df["age"]):
#         errors.append("Column age must be numeric")

#     print(" Data validation passed")
#     return True       

"""
src/data/validate.py
=====================
بوابة جودة (Quality Gate) — يتأكد إن الداتا النظيفة (الناتجة من
transform.py) تستوفي قواعد أساسية قبل ما نثق فيها كفاية نحفظها أو
نرفعها BigQuery.

الفرق عن transform.py: هنا أي فحص فشل = exception بتوقف الـ pipeline
بالكامل. مفيش "تحذير وكمّل" هنا — القرار إن الداتا مش موثوقة قرار
نهائي في المرحلة دي.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """
    Exception مخصص (custom) بدل ما نستخدم Exception عادي.
    الفايدة: أي كود بينادي validate() يقدر يمسك DataValidationError
    تحديدًا (except DataValidationError) من غير ما يمسك أي خطأ تاني
    غلط حصل بالصدفة (زي KeyError من باج في الكود نفسه) ويتعامل معاه
    وكأنه مشكلة بيانات.
    """
    pass


# ── قواعد الفحص (Business Rules) ─────────────────────────────────────────
# نفس مبدأ transform.py: دي قواعد عمل، مش إعدادات بيئة، فمكانها هنا
# جوه الكود مش config.yaml

EXPECTED_COLUMNS = {
    "customer_id", "surname", "credit_score", "geography", "gender",
    "age", "tenure", "balance", "num_of_products", "has_cr_card",
    "is_active_member", "estimated_salary", "exited", "complain",
    "satisfaction_score", "card_type", "point_earned",
}

# القيم المسموحة لكل عمود categorical — أي قيمة برّه القايمة دي
# معناها فيه مشكلة حقيقية في المصدر (خطأ إدخال، أو تصنيف جديد ظهر
# بدون علمنا) لازم نوقف ونتأكد منها يدويًا
ALLOWED_CATEGORIES = {
    "geography": {"France", "Germany", "Spain"},
    "gender": {"Male", "Female"},
    "card_type": {"DIAMOND", "GOLD", "SILVER", "PLATINUM"},
}

# نطاق معقول لنسبة الـ churn الكلية. لو النسبة طلعت 0% أو 100% فجأة،
# ده مش "بيانات نظيفة أكتر" — ده على الأغلب خطأ جسيم في مصدر البيانات
# أو في كود التنظيف نفسه
MIN_PLAUSIBLE_CHURN_RATE = 0.01
MAX_PLAUSIBLE_CHURN_RATE = 0.99


# ── فحوصات فردية (كل واحدة بمسؤولية واحدة بس) ────────────────────────────
def validate_schema(df: pd.DataFrame) -> None:
    """كل الأعمدة المتوقعة موجودة فعليًا — ولا واحد ضاع في التنظيف."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing expected columns: {missing}")


def validate_no_nulls(df: pd.DataFrame) -> None:
    """
    مفيش أي قيمة فارغة بعد التنظيف. لاحظ الفرق عن transform.py:
    هناك كنا "نملأ" الفارغ، وهنا بنتأكد إن مفيش فارغ أصلاً فضل.
    """
    null_counts = df.isnull().sum()
    if null_counts.any():
        raise DataValidationError(
            f"Unexpected nulls found after cleaning:\n{null_counts[null_counts > 0]}"
        )


def validate_no_duplicates(df: pd.DataFrame) -> None:
    """كل customer_id فريد — التكرار المفروض اتشال في transform.py."""
    dup_count = df["customer_id"].duplicated().sum()
    if dup_count > 0:
        raise DataValidationError(f"{dup_count} duplicate customer_id rows found")


def validate_categories(df: pd.DataFrame) -> None:
    """كل قيمة في الأعمدة categorical من ضمن القيم المسموحة والمعروفة."""
    for col, allowed in ALLOWED_CATEGORIES.items():
        actual = set(df[col].unique())
        unexpected = actual - allowed
        if unexpected:
            raise DataValidationError(f"Unexpected values in '{col}': {unexpected}")


def validate_target_distribution(df: pd.DataFrame) -> None:
    """
    نسبة الـ churn الكلية معقولة إحصائيًا. لو طلعت خارج النطاق،
    غالبًا فيه خطأ في مرحلة سابقة (extract غلط، أو join مكرر...).
    """
    churn_rate = df["exited"].mean()
    if not (MIN_PLAUSIBLE_CHURN_RATE <= churn_rate <= MAX_PLAUSIBLE_CHURN_RATE):
        raise DataValidationError(
            f"Churn rate {churn_rate:.2%} is outside plausible bounds "
            f"({MIN_PLAUSIBLE_CHURN_RATE:.0%}-{MAX_PLAUSIBLE_CHURN_RATE:.0%})"
        )


# ── نقطة الدخول الوحيدة اللي pipeline.py هينادي عليها ─────────────────────
def validate(df: pd.DataFrame) -> None:
    """
    IN  : df نظيف من transform.py
    OUT : لا شيء لو كل الفحوصات نجحت. يرمي DataValidationError فورًا
          عند أول فحص فاشل (مش بيكمل يجمع كل الأخطاء — أول مشكلة
          توقف كل حاجة، لأن الاستمرار بعدها معناه بناء فوق أساس مش موثوق).
    """
    logger.info("Running data validation checks...")

    validate_schema(df)
    validate_no_nulls(df)
    validate_no_duplicates(df)
    validate_categories(df)
    validate_target_distribution(df)

    logger.info("All validation checks passed.")


# ── MANUAL TEST BLOCK ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from extract import extract
    from transform import clean

    raw_df = extract()
    clean_df = clean(raw_df)

    try:
        validate(clean_df)
        print("Validation PASSED — الداتا جاهزة للحفظ/الرفع")
    except DataValidationError as e:
        print(f"Validation FAILED: {e}")