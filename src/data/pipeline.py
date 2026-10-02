# """
# src/data/pipeline.py
# =====================
# ORCHESTRATION — الملف الوحيد اللي المفروض يتنفذ مباشرة في الإنتاج.

# مش فيه أي منطق تنظيف أو تحويل بنفسه — بس بينادي extract() ثم clean()
# ثم load_to_parquet() بالترتيب الصح. ده الملف اللي هيتنادى من الـ FastAPI
# endpoint (src/api/main.py) أو من Airflow DAG (dags/churn_pipeline_dag.py)
# — كلاهما هيكتب سطر واحد بس: run_pipeline().
# """

# import logging

# from extract import extract
# from transform import clean
# from load import load_to_parquet, load_to_bigquery

# logger = logging.getLogger(__name__)


# def setup_logging() -> None:
#     """
#     إعداد الـ logging الوحيد لكل المشروع — بيتنفذ مرة واحدة بس هنا.
#     كل الملفات التانية (extract, transform, load) عندها getLogger(__name__)
#     بس، فلما تتنادى من هنا، كل رسائلهم هتطبع بنفس الشكل الموحّد ده.
#     """
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#     )


# def run_pipeline(upload_to_bigquery: bool = False) -> None:
#     """
#     IN  : upload_to_bigquery — علم (flag) اختياري لتفعيل رفع BigQuery.
#           افتراضيًا False عشان تقدر تجرب الـ pipeline محليًا من غير ما
#           تحتاج credentials جاهزة كل مرة.
#     OUT : لا شيء — الدالة دي "فعل" (orchestration)، مش حساب قيمة.

#     لاحظ الاسم "run_pipeline" مش "main" — عشان لو استوردته من
#     src/api/main.py، الاسم يوضّح دوره حتى برّه الملف ده.
#     """
#     logger.info("=" * 50)
#     logger.info("Pipeline started")

#     raw_df = extract()
#     clean_df = clean(raw_df)
#     parquet_path = load_to_parquet(clean_df)

#     if upload_to_bigquery:
#         load_to_bigquery(parquet_path)

#     logger.info("Pipeline finished successfully")
#     logger.info("=" * 50)


# if __name__ == "__main__":
#     setup_logging()
#     run_pipeline(upload_to_bigquery=True)  # غيّرها True لما تكون جاهز ترفع BigQuery كمان

"""
src/data/pipeline.py
=====================
ORCHESTRATION — الملف الوحيد اللي المفروض يتنفذ مباشرة في الإنتاج.

مش فيه أي منطق تنظيف أو تحويل بنفسه — بس بينادي extract() ثم clean()
ثم load_to_parquet() بالترتيب الصح. ده الملف اللي هيتنادى من الـ FastAPI
endpoint (src/api/main.py) أو من Airflow DAG (dags/churn_pipeline_dag.py)
— كلاهما هيكتب سطر واحد بس: run_pipeline().
"""

import logging

from extract import extract
from transform import clean
from validate import validate, DataValidationError
from load import load_to_parquet, load_to_bigquery

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """
    إعداد الـ logging الوحيد لكل المشروع — بيتنفذ مرة واحدة بس هنا.
    كل الملفات التانية (extract, transform, load) عندها getLogger(__name__)
    بس، فلما تتنادى من هنا، كل رسائلهم هتطبع بنفس الشكل الموحّد ده.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_pipeline(upload_to_bigquery: bool = False) -> None:
    """
    IN  : upload_to_bigquery — علم (flag) اختياري لتفعيل رفع BigQuery.
          افتراضيًا False عشان تقدر تجرب الـ pipeline محليًا من غير ما
          تحتاج credentials جاهزة كل مرة.
    OUT : لا شيء — الدالة دي "فعل" (orchestration)، مش حساب قيمة.

    لاحظ الاسم "run_pipeline" مش "main" — عشان لو استوردته من
    src/api/main.py، الاسم يوضّح دوره حتى برّه الملف ده.
    """
    logger.info("=" * 50)
    logger.info("Pipeline started")

    raw_df = extract()
    clean_df = clean(raw_df)

    # بوابة الجودة: لو فشلت، نسجّل رسالة واضحة في اللوج ونوقف فورًا —
    # قبل ما نوصل لـ load_to_parquet أو load_to_bigquery خالص.
    # "raise" من غير أي حاجة بعدها = يعيد رفع نفس الـ exception الأصلي
    # (بنفس الـ traceback)، مش يخفيه أو يستبدله بحاجة تانية.
    try:
        validate(clean_df)
    except DataValidationError:
        logger.error("Data validation failed — pipeline halted before load stage.")
        raise

    parquet_path = load_to_parquet(clean_df)

    if upload_to_bigquery:
        load_to_bigquery(parquet_path)

    logger.info("Pipeline finished successfully")
    logger.info("=" * 50)


if __name__ == "__main__":
    setup_logging()
    run_pipeline(upload_to_bigquery=True)  # غيّرها True لما تكون جاهز ترفع BigQuery كمان