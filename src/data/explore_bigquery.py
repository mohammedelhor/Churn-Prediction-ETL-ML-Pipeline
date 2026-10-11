"""
src/data/explore_bigquery.py
=============================
نفس EDA بتاع explore_sql.py بالظبط، لكن بيشتغل على BigQuery الحقيقي
بدل DuckDB المحلي. قارن الملف ده بـ explore_sql.py وشوف: نص الـ SQL
نفسه تقريبًا حرفيًا — الفرق كله في "إزاي بنتصل ونجيب النتيجة"، مش في
لغة الاستعلام نفسها.
"""

from google.cloud import bigquery
from src.data.paths import load_config


def get_bq_client_and_table(config: dict):
    """
    IN  : dict الإعدادات من config.yaml
    OUT : (client, table_id) — عميل BigQuery موصّل، واسم الجدول الكامل

    نفس منطق load_to_bigquery في load.py بالظبط — نعيد استخدامه هنا
    بدل ما نكرره.
    """
    bq_config = config["bigquery"]
    client = bigquery.Client(project=bq_config["project_id"])
    table_id = f"{bq_config['project_id']}.{bq_config['dataset']}.{bq_config['table']}"
    return client, table_id


CATEGORICAL_COLS = [
    "geography", "gender", "card_type", "num_of_products",
    "has_cr_card", "is_active_member", "complain", "satisfaction_score",
]

NUMERIC_COLS = [
    "credit_score", "age", "balance", "estimated_salary", "point_earned", "tenure",
]


def churn_rate_by_category(client, table_id: str, col: str):
    """
    الفرق الوحيد عن نسخة DuckDB: FROM `{table_id}` بدل read_parquet('...'),
    وbackticks (`) حوالين اسم الجدول — مطلوبة في BigQuery لأن أسماء
    المشاريع فيها نقط وشرطات مش أسماء SQL عادية.
    """
    query = f"""
        SELECT {col},
               COUNT(*) AS total_customers,
               SUM(exited) AS churned,
               ROUND(SUM(exited) * 100.0 / COUNT(*), 2) AS churn_rate_pct
        FROM `{table_id}`
        GROUP BY {col}
        ORDER BY churn_rate_pct DESC
    """
    # client.query(query) بترجع job (زي load_to_bigquery بالظبط) —
    # .to_dataframe() بتستنى النتيجة وتحوّلها pandas DataFrame في خطوة واحدة
    return client.query(query).to_dataframe()


def avg_by_churn_status(client, table_id: str, col: str):
    query = f"""
        SELECT exited,
               ROUND(AVG({col}), 2) AS avg_{col},
               COUNT(*) AS customers
        FROM `{table_id}`
        GROUP BY exited
    """
    return client.query(query).to_dataframe()


if __name__ == "__main__":
    config = load_config()
    client, table_id = get_bq_client_and_table(config)

    print(f"Querying: {table_id}\n")

    print("=" * 60)
    print("CATEGORICAL - نسبة churn لكل فئة (من BigQuery)")
    print("=" * 60)
    for col in CATEGORICAL_COLS:
        print(f"\n--- {col} ---")
        print(churn_rate_by_category(client, table_id, col))

    print("\n" + "=" * 60)
    print("NUMERIC - متوسط القيمة حسب حالة churn (من BigQuery)")
    print("=" * 60)
    for col in NUMERIC_COLS:
        print(f"\n--- {col} ---")
        print(avg_by_churn_status(client, table_id, col))