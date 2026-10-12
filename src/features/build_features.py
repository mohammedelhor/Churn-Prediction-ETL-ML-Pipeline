"""
src/features/build_features.py
================================
FEATURE ENGINEERING — ياخد الـ parquet النظيف والمتحقق منه (بعد validate.py)
ويرجّع نسخة جاهزة للتدريب: أعمدة غير مفيدة اتشالت، categorical
اتحوّلت one-hot، وفيها features جديدة مبنية على اكتشافات الـ EDA.

لاحظ الاستيراد: from src.data.paths — مش from paths زي باقي الملفات.
هذا الملف لازم يتشغّل من روت المشروع بـ:
    python -m src.features.build_features
مش بتشغيله المباشر (python src/features/build_features.py).
"""

import logging
import pandas as pd

from src.data.paths import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)


# ── أعمدة مش مفيدة كـ features (معرّفات، مش سمات تنبؤية) ──────────────────
DROP_COLS = ["customer_id", "surname"]

# الأعمدة categorical اللي هتتحول one-hot
ONE_HOT_COLS = ["geography", "gender", "card_type"]


def get_processed_parquet_path():
    config = load_config()
    return PROJECT_ROOT / config["data"]["processed_dir"] / config["data"]["processed_filename"]


def get_features_output_path():
    config = load_config()
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    return processed_dir / "customer_churn_features.parquet"


def get_feature_columns_path():
    """
    مسار ملف بيحفظ أسماء أعمدة الـ features بالظبط (من غير target)،
    بنفس الترتيب اللي اتدرب عليه الموديل — الـ API محتاجها عشان تبني
    نفس شكل الأعمدة لأي عميل جديد وقت التنبؤ.
    """
    config = load_config()
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    return processed_dir / "feature_columns.json"


# ── هندسة features جديدة مبنية على اكتشافات الـ EDA ────────────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    IN  : df نظيف (بعد drop الأعمدة غير المفيدة، قبل one-hot)
    OUT : نفس df + أعمدة جديدة مبنية على ملاحظات EDA فعلية، مش تخمين

    is_zero_balance:
        36.2% من العملاء رصيدهم صفر بالظبط — نسبة كبيرة جدًا تستاهل
        عمود منفصل، لأن "صفر" هنا مش قيمة عادية على الطيف، هي على
        الأغلب فئة مختلفة تمامًا من العملاء (زي اللي عندهم حساب واحد
        بس من غير رصيد فعلي فيه).

    balance_per_product:
        num_of_products مينفعش يبقى صفر (المدى الفعلي 1-4)، فالقسمة
        آمنة من غير خطر ZeroDivisionError. الفكرة: "متوسط الرصيد لكل
        منتج" ممكن يكون إشارة أوضح من balance وnum_of_products لوحدهم،
        خصوصًا بعد ما شفنا في EDA إن العلاقة بينهم مش خطية.
    """
    df = df.copy()

    df["is_zero_balance"] = (df["balance"] == 0).astype("int8")
    df["balance_per_product"] = (df["balance"] / df["num_of_products"]).round(2)

    logger.info(
        f"Added engineered features: is_zero_balance "
        f"({df['is_zero_balance'].sum()} customers), balance_per_product"
    )

    return df


# ── الدالة الرئيسية ──────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    IN  : df نظيف ومتحقق منه (خرج transform.py بعد ما عدّى validate.py)
    OUT : df جاهز للتدريب — من غير معرّفات، categorical محوّلة one-hot،
          مع features جديدة

    ⚠️ ملاحظة موثّقة عن complain:
    العمود ده بيتسبب في نسبة churn تقريبًا 99.5% لو =1 مقابل 0.05%
    لو =0 — علاقة أقوى بكتير من أي feature تاني. القرار اتاخد إننا
    نسيبه كـ feature (مش نشيله)، لكن ده معناه احتمال حقيقي إن الموديل
    هيعتمد عليه بشكل شبه كامل (data leakage محتمل)، لأن الشكوى غالبًا
    بتحصل كنتيجة لقرار المغادرة مش سبب له. لو الأداء طلع عالي جدًا
    بشكل مريب وقت التقييم (evaluate.py)، ده أول مكان نرجع نشك فيه.
    """
    df = df.drop(columns=DROP_COLS, errors="ignore")

    df = add_engineered_features(df)

    # pd.get_dummies بيحول كل عمود categorical لعدة أعمدة 0/1، واحد
    # لكل فئة. drop_first=True بيشيل فئة واحدة من كل عمود (بتبقى
    # ممثّلة ضمنيًا لما الباقي كلهم يبقوا صفر) — بيقلل التكرار
    # (multicollinearity) لو استخدمت موديلات خطية بعدين
    df = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=True)

    logger.info(f"Feature matrix shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def prepare_single_customer(raw: dict, feature_columns: list) -> pd.DataFrame:
    """
    IN  : dict خام لعميل واحد — نفس أعمدة العميل الخام (credit_score,
          geography, gender, age...)، من غير customer_id/surname/complain
          (complain مستبعد أصلاً من الموديل النهائي، وأي حال معلومة
          مش متاحة قبل ما العميل يتخذ قراره فعليًا)
    OUT : DataFrame بصف واحد، بنفس أعمدة وترتيب feature_columns بالظبط

    هنا حل مشكلة "training-serving skew": get_dummies على صف واحد
    بس مش هيعرف يبني كل أعمدة one-hot اللي الموديل يتوقعها (لو العميل
    من فرنسا مثلاً، مفيش قيم تانية في نفس الصف تقارن بيها لبناء
    geography_Germany أو geography_Spain). الحل: نبني اللي نقدر، وبعدين
    reindex على نفس أعمدة التدريب بالظبط — أي عمود ناقص يتملى صفر
    تلقائيًا، وأي عمود زيادة (مش متوقع) يتشال.
    """
    df = pd.DataFrame([raw])
    df = add_engineered_features(df)

    # ⚠️ ليه مش pd.get_dummies(..., drop_first=True) هنا؟
    # على صف واحد، العمود فيه فئة واحدة بس، فـ drop_first=True بيشيل
    # الفئة دي نفسها! النتيجة: كل عميل (ألماني، ذكر، GOLD...) بيتحوّل
    # لنفس الشكل الافتراضي (فرنسا/أنثى/DIAMOND) من غير أي خطأ ظاهر.
    # اكتشفناه باختبار tests/test_serving_consistency.py.
    # الحل: نبني عمود الـ one-hot يدويًا بنفس اسم عمود التدريب
    # (مثلًا geography_Germany)، ولو الاسم مش في feature_columns يبقى
    # العميل من الفئة الأساسية اللي اتشالت وقت التدريب، فيفضل صفر.
    for col in ONE_HOT_COLS:
        value = df.at[0, col]
        df = df.drop(columns=col)
        dummy_name = f"{col}_{value}"
        if dummy_name in feature_columns:
            df[dummy_name] = 1

    # أي عمود ناقص (فئات التدريب التانية) يتملى صفر، وأي زيادة تتشال
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


# ── MANUAL TEST BLOCK ────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    input_path = get_processed_parquet_path()
    clean_df = pd.read_parquet(input_path)

    features_df = build_features(clean_df)

    output_path = get_features_output_path()
    features_df.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")

    # نحفظ أسماء أعمدة الـ features بالظبط (من غير target وcomplain) —
    # لازم نفس الاستبعاد بالظبط اللي حصل في train.py للموديل النهائي
    # (XGBoost)، وإلا الـ API هتبني شكل أعمدة مختلف عن اللي الموديل
    # المحفوظ فعليًا يتوقعه
    feature_columns = [c for c in features_df.columns if c not in ("exited", "complain")]
    with open(get_feature_columns_path(), "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)

    print(f"Saved features to: {output_path}")
    print(f"Saved feature columns list to: {get_feature_columns_path()}")
    print(features_df.head())
    print(f"Shape: {features_df.shape}")