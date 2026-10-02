"""
dags/churn_pipeline_dag.py
============================
DAG بينادي run_pipeline() من src/data/pipeline.py مباشرة — نفس الكود
اللي بنشغّله يدويًا، من غير ما نكرر أي منطق هنا.

ملاحظة معمارية مهمة: run_pipeline() في pipeline.py متعمدين متنادوش
setup_logging() جواها (فاكر ليه: setup_logging لازم تتنفذ مرة واحدة
بس، من نقطة دخول واحدة). هنا Airflow نفسه هو نقطة الدخول، وهو أصلاً
بيظبط الـ logging الخاص بيه لكل task — فمعندناش أي تعارض، ونشوف لوج
الـ pipeline بالكامل من واجهة Airflow مباشرة.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data.pipeline import run_pipeline

default_args = {
    "owner": "aura-labs",
    "retries": 1,
}

with DAG(
    dag_id="churn_etl_pipeline",
    description="Extract -> Transform -> Validate -> Load للبيانات، يوميًا",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,  # ما ينفذش تشغيلات "فايتة" من تواريخ قديمة، يبدأ من دلوقتي بس
    tags=["churn", "etl"],
) as dag:

    run_etl_task = PythonOperator(
        task_id="run_etl_pipeline",
        python_callable=run_pipeline,
        # upload_to_bigquery=False افتراضيًا هنا كمان — فاكر مشكلة
        # الـ credentials جوه Docker اللي اتكلمنا عنها؟ لو ضبطناها
        # بعدين، هنا بالظبط المكان اللي نغيّر فيه القيمة لـ True
        op_kwargs={"upload_to_bigquery": False},
    )