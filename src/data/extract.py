"""
src/data/extract.py
====================
EXTRACT stage — الخطوة الأولى في الـ pipeline (Extract -> Transform -> Load).

المسؤولية الوحيدة لهذا الملف: جلب البيانات الخام من مصدرها، من غير أي
تنظيف أو تعديل. لو حسّيت إنك محتاج تضيف "if" بيتحقق من قيمة أو يمسح صف،
فده مكانه transform.py مش هنا.
"""

# كل ما كان مكرر هنا (PROJECT_ROOT, CONFIG_PATH, load_config) بقى في
# ملف مشترك واحد — paths.py — وباقي الملفات بتستورد منه
from pathlib import Path
import pandas as pd
import logging

from src.data.paths import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)


# ── EXTRACT FUNCTION ─────────────────────────────────────────────────────
def extract(path: Path | None = None) -> pd.DataFrame:
    """
    IN  : path اختياري. لو مبعتّهوش، هيتبني تلقائيًا من config.yaml.
          (مرونة: تقدر تنادي extract() عادي في المشروع الحقيقي،
           أو extract(ملف_تجريبي) لو بتكتب unit test)
    OUT : pandas DataFrame يحتوي البيانات الخام زي ما هي، من غير أي تعديل.

    ليه لازم يبقى "غبي" عمدًا؟
    لو حطينا هنا أي منطق تنظيف، وبعدين غيّرنا المصدر (من CSV لـ database
    مثلاً)، هنضطر نكرر منطق التنظيف في مكانين. extract يجيب بس، خلاص.
    """
    if path is None:
        # لو محددش مسار، ابني الافتراضي من الإعدادات
        config = load_config()
        path = PROJECT_ROOT / config["data"]["raw_dir"] / config["data"]["raw_filename"]

    logger.info(f"Extracting from {path}")
    df = pd.read_csv(path)
    logger.info(f"Extracted shape: {df.shape}")

    return df


# ── MANUAL TEST BLOCK ────────────────────────────────────────────────────
# هذا الجزء بيتنفذ بس لو شغّلت الملف مباشرة (python extract.py)،
# مش لو استوردته من ملف تاني (import extract). ده بيديك طريقة سريعة
# تتأكد إن الدالة شغالة من غير ما تكتب pipeline كامل أو test رسمي.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # هنا بس مسموح، لأن الملف بيتشغل standalone
    df = extract()
    print(df.head())
    print(f"Shape: {df.shape}")