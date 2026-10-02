"""
src/api/main.py
=================
FastAPI app — endpoint /predict بيستخدم موديل XGBoost النهائي، بعتبة
0.40 (القرار اللي اتاخد في evaluate.py بالتحليل الحدّي، مش العتبة
الافتراضية 0.5).
"""

import json
import joblib
from fastapi import FastAPI

from src.data.paths import PROJECT_ROOT
from src.features.build_features import prepare_single_customer, get_feature_columns_path
from src.api.schemas import CustomerInput, PredictionOutput

# القرار المعتمد من evaluate.py — أفضل توازن Recall/Precision لهدف
# المشروع (اكتشاف أكبر عدد ممكن من العملاء المهددين بالمغادرة)
DECISION_THRESHOLD = 0.40

app = FastAPI(title="Churn Prediction API")

# الموديل وقايمة الأعمدة بيتحمّلوا مرة واحدة بس وقت بدء تشغيل السيرفر
# (startup) — مش مع كل طلب. تحميل موديل من القرص عملية بطيئة نسبيًا،
# وتكرارها مع كل request هيبوّظ سرعة الـ API بالكامل
_model = None
_feature_columns = None


@app.on_event("startup")
def load_artifacts():
    global _model, _feature_columns

    model_path = PROJECT_ROOT / "joblib" / "churn_model_xgboost.joblib"
    _model = joblib.load(model_path)

    with open(get_feature_columns_path(), encoding="utf-8") as f:
        _feature_columns = json.load(f)


@app.get("/health")
def health():
    """فحص سريع إن السيرفر شغال والموديل اتحمّل صح — تستخدمها Docker/monitoring."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    """
    IN  : بيانات عميل واحد (CustomerInput، Pydantic بيتحقق منها تلقائيًا)
    OUT : احتمالية المغادرة + القرار النهائي بناءً على العتبة المعتمدة
    """
    row = prepare_single_customer(customer.model_dump(), _feature_columns)

    probability = float(_model.predict_proba(row)[0][1])
    will_churn = probability >= DECISION_THRESHOLD

    return PredictionOutput(
        churn_probability=round(probability, 4),
        will_churn=will_churn,
        threshold_used=DECISION_THRESHOLD,
    )