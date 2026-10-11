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
from src.data.paths import PROJECT_ROOT, load_config


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