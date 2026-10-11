"""
src/data/load.py
=================
LOAD stage — الخطوة الأخيرة في الـ pipeline (Extract -> Transform -> Load).

المسؤولية: ياخد DataFrame نظيف (من transform.py) ويحفظه في وجهته
النهائية. فيه وجهتين هنا: ملف parquet محلي (دايمًا بيحصل)، وBigQuery
(اختياري، بيحصل بس لو طلبته صراحة — لأنه محتاج credentials مش متوفرة
افتراضيًا).
"""

# كل ما كان مكرر هنا بقى في paths.py، وده بيستورد منه بدل ما يكرره
from pathlib import Path
import logging
import pandas as pd

from src.data.paths import PROJECT_ROOT, load_config, CONFIG_PATH

logger = logging.getLogger(__name__)


# ── PARQUET LOADER ────────────────────────────────────────────────────────
def load_to_parquet(df: pd.DataFrame, path: Path | None = None) -> Path:
    """
    IN  : df نظيف من transform.py، وpath اختياري (زي extract تمامًا،
          لو مبعتّهوش هيتبني من config.yaml)
    OUT : المسار اللي اتحفظ فيه الملف فعليًا (مفيد لو load_to_bigquery
          هيحتاج يقرا نفس الملف بعد كده)

    ليه parquet مش CSV؟
    - columnar format: قراءة عمود واحد بس مش محتاجة تقرا الملف كله
    - بيحفظ الـ dtype جوّاه (CSV بيحوّل كل حاجة لـ text ويرميها)
    - ضغط أعلى بكتير — شفنا الفرق بنفسك (950KB → 295KB تقريبًا)

    engine="pyarrow" + compression="snappy":
    pyarrow هو الأسرع والأكتر توافق مع BigQuery ومعظم أدوات الـ data
    engineering. snappy بيوازن بين سرعة الضغط/فك الضغط وحجم الملف
    (مش أصغر ضغط ممكن، لكنه الأنسب لملفات بتتقرا كتير).
    """
    if path is None:
        config = load_config()
        processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
        processed_dir.mkdir(parents=True, exist_ok=True)  # ينشئ المجلد لو مش موجود
        path = processed_dir / config["data"]["processed_filename"]
    else:
        # لو حد بعت string بدل Path (زي "output.parquet")، نحولها Path
        # هنا مرة واحدة، بدل ما نفترض إن كل نداء للدالة هيبعت Path سليم
        path = Path(path)

    df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")

    size_kb = path.stat().st_size / 1024
    logger.info(f"Wrote parquet: {path} ({size_kb:.1f} KB)")

    return path


# ── BIGQUERY LOADER (اختياري) ─────────────────────────────────────────────
def load_to_bigquery(parquet_path: Path, config_path: Path = CONFIG_PATH) -> None:
    """
    IN  : مسار ملف parquet (المفروض تكون النتيجة اللي رجعتها load_to_parquet)
    OUT : لا شيء (side effect بس: البيانات بتترفع فعليًا على BigQuery)

    الشروط قبل ما تشتغل:
    1) pip install google-cloud-bigquery
    2) عندك service account key، ومساره محطوط في متغير بيئة اسمه
       GOOGLE_APPLICATION_CREDENTIALS
    3) project_id و dataset و table متعرّفين في config.yaml

    import مكتبة google.cloud.bigquery جوه الدالة نفسها، مش فوق مع
    باقي الـ imports — ليه؟ عشان لو حد شغّل الملف ده وعنده المكتبة دي
    مش متثبتة (لأنه مش محتاجها دلوقتي)، باقي الملف (parquet loading)
    يفضل شغال عادي من غير ما يطلعله ImportError من الأول.
    """
    from google.cloud import bigquery

    config = load_config(config_path)
    bq_config = config["bigquery"]

    project_id = bq_config["project_id"]
    dataset = bq_config["dataset"]
    table = bq_config["table"]

    if not project_id:
        # فحص واضح بدل ما نسيب المكتبة ترمي error غامض لو project_id فاضي
        raise ValueError("bigquery.project_id لسه فاضي في config.yaml")

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset}.{table}"

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        # WRITE_TRUNCATE: كل تشغيلة جديدة بتستبدل بيانات الجدول بالكامل.
        # (لو عايز تضيف بدل ما تستبدل، تغيّرها لـ WRITE_APPEND)
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    with open(parquet_path, "rb") as f:
        job = client.load_table_from_file(f, table_id, job_config=job_config)
    job.result()  # ينتظر لحد ما الـ job يخلص فعليًا قبل ما يكمل

    table_ref = client.get_table(table_id)
    logger.info(f"Loaded {table_ref.num_rows} rows into {table_id}")


# ── MANUAL TEST BLOCK ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.data.extract import extract
    from src.data.transform import clean

    raw_df = extract()
    clean_df = clean(raw_df)
    parquet_path = load_to_parquet(clean_df)

    print(f"Saved to: {parquet_path}")

    # load_to_bigquery(parquet_path)  # فعّلها لما تجهز الـ credentials