"""
src/data/transform.py
======================
TRANSFORM stage — الخطوة التانية في الـ pipeline (Extract -> Transform -> Load).

المسؤولية الوحيدة لهذا الملف: ياخد DataFrame خام (من extract.py)، ويرجّع
DataFrame نظيف وجاهز للحفظ. الملف ده مبيعرفش أي حاجة عن مصدر البيانات
(CSV ولا database ولا API) ولا عن وجهتها (parquet ولا BigQuery) — دي
مسؤولية extract.py و load.py. الفصل ده بيخليك تقدر تغيّر المصدر أو
الوجهة من غير ما تلمس سطر تنظيف واحد هنا.
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────
import re                     # للتعامل مع الأنماط النصية (regex) - هنستخدمها في تحويل أسماء الأعمدة
import logging
import pandas as pd

logger = logging.getLogger(__name__)   # نفس المبدأ: getLogger بس، الإعداد الفعلي في pipeline.py


# ── BUSINESS RULES (مش config.yaml — ودي النقطة المهمة) ──────────────────
# ليه القيم دي هنا في الكود، مش في config.yaml زي المسارات؟
#
# config.yaml = إعدادات "بيئة التشغيل" (environment): مسارات، أسماء ملفات،
#   حاجات بتختلف من جهاز لجهاز أو من بيئة (dev/production) لبيئة.
#
# القيم اللي تحت دي = "قواعد عمل" (business logic): جزء من تعريف
#   "البيانات الصحيحة" نفسه. تغييرها معناه قرار عمل جديد (مثلاً:
#   "دلوقتي بنقبل عملاء أقل من 18 سنة")، مش مجرد تغيير بيئة تشغيل.
#   عشان كده صح تفضل هنا جوه الكود، قريبة من المنطق اللي بتتحكم فيه.

# نطاقات القيم المنطقية لكل عمود رقمي — أي قيمة برّه النطاق ده بتتسجل
# كتحذير (مش بتتمسح تلقائيًا، خليك فاكر القرار ده من كلامنا الأول)
VALID_RANGES = {
    "age": (18, 100),
    "credit_score": (300, 850),
    "satisfaction_score": (1, 5),
    "tenure": (0, 10),
}

# الأعمدة اللي قيمتها 0/1 بس (flags) — هنحولها لنوع بيانات موحد
BOOLEAN_LIKE_COLUMNS = ["has_cr_card", "is_active_member", "exited", "complain"]

# خريطة تحويل كل عمود لأصغر نوع بيانات يقدر يستوعب قيمه فعليًا
# (ده اللي وفّر الحجم من ~950KB لـ 295KB في التجربة الأولى)
DTYPE_MAP = {
    "customer_id": "int64",       # identifier، مش كمية — نسيبه int64 دايمًا (مش بيتحسب من الـ range)
    "age": "int8",                 # فعلي: 18-92 → int8 (حتى 127) كفاية، int16 كان زيادة
    "tenure": "int8",
    "credit_score": "int16",       # فعلي: 350-850 → أكبر من حد int8 (127)، محتاج int16
    "num_of_products": "int8",
    "satisfaction_score": "int8",
    "point_earned": "int16",       # فعلي: 119-1000 → أكبر من int8، لكن int16 (حتى 32767) كافي، int32 كان زيادة
    "balance": "float64",
    "estimated_salary": "float64",
}

# الأعمدة النصية اللي قيمها محدودة ومتكررة → category أوفر في المساحة
CATEGORY_COLUMNS = ["geography", "gender", "card_type"]


# ── COLUMN NAME NORMALIZATION ───────────────────────────────────────────
def to_snake_case(name: str) -> str:
    """
    IN  : اسم عمود زي ما هو في الـ CSV الخام، مثلاً "Satisfaction Score"
    OUT : نفس الاسم بصيغة snake_case، مثلاً "satisfaction_score"

    الخطوات (بالترتيب، والترتيب مهم هنا):
    1) نحط _ بين حرف صغير/رقم وحرف كبير جاي بعده مباشرة (لحالة camelCase
       زي "CustomerId" → "Customer_Id")
    2) نستبدل أي مسافة بـ _  (لحالة "Satisfaction Score" → "Satisfaction_Score")
    3) ندمج أي _  متكررة في _ واحدة، احتياطًا لو الخطوتين فوق نتج عنهم
       أكتر من _ ورا بعض في نفس المكان
    4) نحول كل حاجة لحروف صغيرة في الآخر
    """
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = name.replace(" ", "_")
    name = re.sub(r"_+", "_", name)
    return name.lower()


# ── MAIN CLEAN FUNCTION ──────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    IN  : DataFrame خام زي ما خرج من extract() — أعمدة بأسماء الـ CSV
          الأصلية، dtypes افتراضية من pandas (كله int64/float64/object)
    OUT : DataFrame نظيف: أسماء أعمدة موحدة، من غير تكرار، dtypes مضبوطة،
          مع تسجيل أي قيم شاذة كتحذير في اللوج

    df = df.copy() في الأول ليه؟
    عشان الدالة متعدلش في الـ DataFrame الأصلي اللي اتبعت ليها من برّه.
    لو الكود اللي نادى clean() لسه محتاج الـ df الخام لسبب ما (مقارنة،
    تصحيح أخطاء)، النسخة الأصلية هتفضل سليمة.
    """
    df = df.copy()

    # 1) توحيد أسماء الأعمدة
    df.columns = [to_snake_case(c) for c in df.columns]
    logger.info(f"Columns after rename: {list(df.columns)}")

    # 2) حذف row_number — مجرد ترقيم صفوف، مالوش قيمة تحليلية
    #    errors="ignore" يعني: لو العمود مش موجود أصلاً (اتغيّر مصدر
    #    البيانات مثلاً)، الكود ميعملش crash، بس يكمل عادي
    df = df.drop(columns=["row_number"], errors="ignore")

    # 3) إزالة التكرار على customer_id
    #    دي خطوة دفاعية — الداتا الحالية نضيفة، لكن الكود لازم يفترض إن
    #    أي مصدر بيانات حقيقي ممكن يجيب تكرار يومًا ما
    before = len(df)
    df = df.drop_duplicates(subset=["customer_id"], keep="first")
    if before != len(df):
        logger.warning(f"Dropped {before - len(df)} duplicate customer_id rows")

    # 4) التعامل مع القيم الفارغة (نفس مبدأ الدفاع: مفيش فارغ دلوقتي،
    #    لكن الكود لازم يكون جاهز لو ظهر)
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Nulls found:\n{null_counts[null_counts > 0]}")
        for col in df.columns:
            if df[col].isnull().any():
                # عمود رقمي → نملأ بالـ median (أقل تأثرًا بالقيم الشاذة من mean)
                # عمود نصي/تصنيفي → نملأ بالقيمة الأكتر تكرارًا (mode)
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])

    # 5) فحص النطاقات المنطقية — تسجيل تحذير بس، من غير حذف تلقائي.
    #    القرار: أي حذف بيانات لازم يبقى قرار واعي تشوفه في اللوج،
    #    مش حاجة بتحصل جوه الكود من غير ما حد ينتبه
    for col, (lo, hi) in VALID_RANGES.items():
        out_of_range = ~df[col].between(lo, hi)
        if out_of_range.any():
            logger.warning(
                f"{out_of_range.sum()} rows out of range for {col} ({lo}-{hi})"
            )

    # 6) ضبط الـ dtypes حسب الخريطة اللي عرّفناها فوق
    for col, dtype in DTYPE_MAP.items():
        df[col] = df[col].astype(dtype)

    # الأعمدة الـ boolean-like بتتحول لـ int8 (مش bool) عشان:
    # - متوافقة أكتر مع parquet و BigQuery من غير تحويلات إضافية
    # - int8 بياخد مساحة زي bool بالظبط، فمفيش خسارة
    for col in BOOLEAN_LIKE_COLUMNS:
        df[col] = df[col].astype("int8")

    # الأعمدة النصية محدودة القيم → category (بديل موفر للمساحة عن object)
    for col in CATEGORY_COLUMNS:
        df[col] = df[col].astype("category")

    # 7) تنظيف مسافات زيادة في النصوص (قبل التحويل لـ category نهائيًا)
    for col in ["surname"] + CATEGORY_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    # نعيد تحويل أعمدة الـ category تاني بعد الـ strip، لأن astype(str)
    # فوق حوّلها مؤقتًا لـ object
    for col in CATEGORY_COLUMNS:
        df[col] = df[col].astype("category")

    logger.info(f"Post-clean shape: {df.shape}")
    logger.info(f"Dtypes:\n{df.dtypes}")

    return df


# ── MANUAL TEST BLOCK ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # استيراد extract من نفس المجلد، عشان نقدر نختبر transform.py لوحده
    # بسلسلة extract -> clean كاملة، من غير ما نحتاج pipeline.py لسه
    from extract import extract

    raw_df = extract()
    clean_df = clean(raw_df)

    print(clean_df.head())
    print(f"Shape before: {raw_df.shape} | after: {clean_df.shape}")


