# """
# src/data/explore_sql.py
# ========================
# EDA محلي بلغة SQL، باستخدام DuckDB — بديل سريع وخفيف لـ BigQuery
# وقت التطوير المحلي، قبل ما نرفع البيانات فعليًا على السحابة.

# ده ملف تجريبي/استكشافي، مش جزء من الـ pipeline الرسمي (extract/transform/
# load/pipeline) — دوره بس يساعدك تفهم البيانات وتكتشف الأنماط فيها،
# عشان تبني قرارات build_features.py بعد كده على أساس فعلي مش تخمين.
# """

# import duckdb
# from paths import PROJECT_ROOT, load_config


# def get_parquet_path():
#     """
#     IN  : لا شيء
#     OUT : المسار الكامل لملف الـ parquet النظيف، مبني من config.yaml
#           (نفس المنطق اللي استخدمناه في load.py بالظبط)
#     """
#     config = load_config()
#     return PROJECT_ROOT / config["data"]["processed_dir"] / config["data"]["processed_filename"]


# def run_query(sql: str):
#     """
#     IN  : نص استعلام SQL كامل
#     OUT : pandas DataFrame بنتيجة الاستعلام

#     duckdb.sql(sql) بينفذ الاستعلام، و.df() بيحوّل النتيجة لـ
#     DataFrame بدل ما تفضل في صيغة DuckDB الداخلية — أسهل تتعامل معاها
#     وتطبعها أو تحفظها بعد كده.
#     """
#     return duckdb.sql(sql).df()


# if __name__ == "__main__":
#     parquet_path = get_parquet_path()

#     # لاحظ: read_parquet('{parquet_path}') — DuckDB بيقدر "يشوف" ملف
#     # الـ parquet كأنه جدول قاعدة بيانات عادي، من غير أي خطوة تحميل
#     # أو استيراد منفصلة قبلها
#     churn_by_geography = f"""
#         SELECT geography,
#                COUNT(*) AS total_customers,
#                SUM(exited) AS churned,
#                ROUND(SUM(exited) * 100.0 / COUNT(*), 2) AS churn_rate_pct
#         FROM read_parquet('{parquet_path}')
#         GROUP BY geography
#         ORDER BY churn_rate_pct DESC
#     """

#     credit_score_by_status = f"""
#         SELECT exited,
#                ROUND(AVG(credit_score), 1) AS avg_credit_score,
#                ROUND(AVG(balance), 1) AS avg_balance,
#                COUNT(*) AS customers
#         FROM read_parquet('{parquet_path}')
#         GROUP BY exited
#     """

#     print("=== نسبة churn لكل دولة ===")
#     print(run_query(churn_by_geography))

#     print("\n=== متوسط credit_score و balance حسب حالة churn ===")
#     print(run_query(credit_score_by_status))

"""
src/data/explore_sql.py
========================
EDA محلي بلغة SQL (DuckDB) — نسخة معمّمة بالـ loop بدل ما نكتب query
منفصل لكل عمود يدويًا.

المنطق (فاكر كلامنا):
- Categorical/محدود القيم → GROUP BY + نسبة churn لكل فئة
- Numeric مستمر → متوسط القيمة مقارنة بين exited=0 وexited=1

customer_id وsurname مستبعدين عمدًا — الأول معرّف مش سمة تحليلية،
والتاني اسم شخصي مالوش دلالة إحصائية.
"""

import duckdb
from paths import PROJECT_ROOT, load_config


def get_parquet_path():
    config = load_config()
    return PROJECT_ROOT / config["data"]["processed_dir"] / config["data"]["processed_filename"]


# ── تصنيف الأعمدة (نفس الجدول اللي اتفقنا عليه) ───────────────────────────
CATEGORICAL_COLS = [
    "geography", "gender", "card_type", "num_of_products",
    "has_cr_card", "is_active_member", "complain", "satisfaction_score",
]

NUMERIC_COLS = [
    "credit_score", "age", "balance", "estimated_salary", "point_earned", "tenure",
]


# ── الـ templates: query واحد بس لكل نوع، بيتكرر لكل عمود ─────────────────
def churn_rate_by_category(parquet_path, col: str):
    """
    IN  : اسم عمود categorical
    OUT : DataFrame — كل فئة، عدد عملائها، نسبة churn فيها
    """
    query = f"""
        SELECT {col},
               COUNT(*) AS total_customers,
               SUM(exited) AS churned,
               ROUND(SUM(exited) * 100.0 / COUNT(*), 2) AS churn_rate_pct
        FROM read_parquet('{parquet_path}')
        GROUP BY {col}
        ORDER BY churn_rate_pct DESC
    """
    return duckdb.sql(query).df()


def avg_by_churn_status(parquet_path, col: str):
    """
    IN  : اسم عمود numeric
    OUT : DataFrame — متوسط العمود لكل حالة (exited=0 و exited=1)
    """
    query = f"""
        SELECT exited,
               ROUND(AVG({col}), 2) AS avg_{col},
               COUNT(*) AS customers
        FROM read_parquet('{parquet_path}')
        GROUP BY exited
    """
    return duckdb.sql(query).df()


if __name__ == "__main__":
    parquet_path = get_parquet_path()

    print("=" * 60)
    print("CATEGORICAL - نسبة churn لكل فئة")
    print("=" * 60)
    for col in CATEGORICAL_COLS:
        print(f"\n--- {col} ---")
        print(churn_rate_by_category(parquet_path, col))

    print("\n" + "=" * 60)
    print("NUMERIC - متوسط القيمة حسب حالة churn")
    print("=" * 60)
    for col in NUMERIC_COLS:
        print(f"\n--- {col} ---")
        print(avg_by_churn_status(parquet_path, col))

    # سؤال المتابعة اللي ولّده اكتشاف balance فوق: هل الفرق فعليًا
    # مرتبط بـ num_of_products، مش balance لوحده؟
    print("\n" + "=" * 60)
    print("متابعة: avg_balance حسب num_of_products x exited")
    print("=" * 60)
    followup_query = f"""
        SELECT num_of_products, exited,
               ROUND(AVG(balance), 2) AS avg_balance,
               COUNT(*) AS customers
        FROM read_parquet('{parquet_path}')
        GROUP BY num_of_products, exited
        ORDER BY num_of_products, exited
    """
    print(duckdb.sql(followup_query).df())