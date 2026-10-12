"""
src/models/compare.py
=====================
مقارنة الموديلات التلاتة (Random Forest / XGBoost / Logistic Regression)
على نفس بيانات الاختبار بالظبط، وإنتاج الأرقام اللي هتتحط في الـ README.

ليه الملف ده موجود؟
-------------------
evaluate.py بيقيّم موديل واحد بس، ولو حمّل الملف الغلط (مثلًا الـ RF
بدل XGBoost) مفيش حاجة في الناتج بتقولك. هنا بنطبع:
    1) نوع كل موديل اتحمّل فعلًا (type(model).__name__) وتاريخ تعديل
       ملفه، عشان نكتشف أي ملف قديم أو غلط.
    2) جدول واحد فيه الموديلات التلاتة على نفس الـ split.
    3) تحليل العتبة للموديل الأنسب (XGBoost) بنفس منهج التحليل الحدّي.
    4) ملف JSON بالنتايج (reports/model_comparison.json)، عشان أرقام
       الـ README تتنقل من ملف مش من الذاكرة.

التشغيل (من جذر المشروع):
    python -m src.models.compare
"""

import json
import logging
from datetime import datetime

import joblib
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.paths import PROJECT_ROOT
from src.models.evaluate import (
    find_best_threshold_by_marginal_cost,
    get_evaluation_split,
    threshold_analysis,
)
from src.models.train import get_model_output_path

logger = logging.getLogger(__name__)

# اسم العرض → اسم ملف الموديل جوه فولدر joblib/
MODEL_FILES = {
    "Random Forest": "churn_model.joblib",
    "XGBoost": "churn_model_xgboost.joblib",
    "Logistic Regression": "churn_model_logreg.joblib",
    # الموديل المضبوط من tune.py، اختياري: لو الملف مش موجود بيتخطّاه
    "Random Forest (tuned)": "churn_model_tuned.joblib",
}

# أهداف الـ recall للمقارنة العادلة (كام % من السايبين لازم نمسك)
MATCHED_RECALLS = (0.70, 0.75, 0.80)

# العتبات اللي بنقارن عندها. 0.50 = السلوك الافتراضي، 0.40 = اللي شغّال في الـ API
COMPARE_THRESHOLDS = (0.50, 0.40)

REPORT_PATH = PROJECT_ROOT / "reports" / "model_comparison.json"


def compare_models(models: dict, X_test, y_test, thresholds=COMPARE_THRESHOLDS) -> list:
    """
    IN  : dict {اسم: موديل مدرّب}، بيانات الاختبار، وقايمة عتبات
    OUT : list فيها صف لكل موديل: ROC-AUC + recall/precision/f1 عند كل عتبة

    ROC-AUC بيتحسب من الاحتمالات الخام (مش مرتبط بعتبة)، أما
    recall/precision/f1 فبيتحسبوا للفئة 1 (سايب) عند كل عتبة بنختارها.
    """
    rows = []
    for name, model in models.items():
        # احتمال إن العميل يسيب. لازم نفس X_test لكل الموديلات عشان المقارنة عادلة
        proba = model.predict_proba(X_test)[:, 1]
        row = {
            "model": name,
            "class": type(model).__name__,
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "at_threshold": {},
        }
        for t in thresholds:
            pred = (proba >= t).astype(int)
            row["at_threshold"][f"{t:.2f}"] = {
                "recall": float(recall_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred)),
            }
        rows.append(row)
    return rows


def compare_at_matched_recall(models: dict, X_test, y_test, targets=MATCHED_RECALLS) -> list:
    """
    المقارنة العادلة: نثبّت الـ recall (كام % من السايبين عايزين نمسك)
    ونسأل كل موديل: عشان تمسك النسبة دي، precision بتاعك كام؟ وكام عميل
    هنزعجهم من غير لازمة (false alarms)؟

    ليه ده أعدل من مقارنة عند عتبة واحدة؟
    كل موديل بيوزّع احتمالاته بشكل مختلف (XGBoost بـ scale_pos_weight
    بيرفعها). عند نفس العتبة 0.50 هو بيمسك سايبين أكتر لمجرد إن احتمالاته
    أعلى، مش لأنه أحسن. لما نثبّت الـ recall بنشيل أثر العتبة من المقارنة.

    الطريقة: precision_recall_curve بترجع precision وrecall لكل عتبة ممكنة.
    بنختار العتبات اللي recall عندها >= الهدف، وناخد منها أعلى precision.
    """
    rows = []
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, proba)
        # precision وrecall أطول من thresholds بعنصر واحد (نقطة النهاية)، فنقصّها
        precision, recall = precision[:-1], recall[:-1]
        entry = {"model": name, "targets": {}}
        for target in targets:
            ok = recall >= target
            if not ok.any():
                entry["targets"][f"{target:.2f}"] = None  # الموديل مستحيل يوصل للهدف
                continue
            idx = int(precision[ok].argmax())          # أعلى precision وسط المسموح
            thr = float(thresholds[ok][idx])
            pred = proba >= thr
            tp = int((pred & (y_test.to_numpy() == 1)).sum())
            fp = int(pred.sum()) - tp
            entry["targets"][f"{target:.2f}"] = {
                "threshold": thr,
                "recall": float(recall[ok][idx]),
                "precision": float(precision[ok][idx]),
                "false_alarms": fp,
            }
        rows.append(entry)
    return rows


def print_matched_recall(rows: list, targets=MATCHED_RECALLS) -> None:
    """جدول: لكل هدف recall، كل موديل بيدفع كام false alarm."""
    for target in targets:
        print(f"\nلو عايزين نمسك {target:.0%} من السايبين على الأقل:")
        print(f"{'Model':<24}{'Threshold':>10}{'Precision':>11}{'False alarms':>14}")
        print("-" * 59)
        for row in rows:
            m = row["targets"][f"{target:.2f}"]
            if m is None:
                print(f"{row['model']:<24}{'غير ممكن':>35}")
            else:
                print(
                    f"{row['model']:<24}{m['threshold']:>10.2f}"
                    f"{m['precision']:>11.2%}{m['false_alarms']:>14}"
                )


def print_comparison(rows: list, thresholds=COMPARE_THRESHOLDS) -> None:
    """بيطبع الجدول: صف لكل (موديل، عتبة) عشان يتقري بسهولة."""
    print(f"{'Model':<22}{'ROC-AUC':>9}{'Thresh':>8}{'Recall':>9}{'Precision':>11}{'F1':>8}")
    print("-" * 67)
    for row in rows:
        for t in thresholds:
            m = row["at_threshold"][f"{t:.2f}"]
            print(
                f"{row['model']:<22}{row['roc_auc']:>9.4f}{t:>8.2f}"
                f"{m['recall']:>9.2%}{m['precision']:>11.2%}{m['f1']:>8.2%}"
            )
        print()


def load_all_models() -> dict:
    """
    بيحمّل كل موديل ويطبع نوعه وتاريخ تعديل ملفه.
    لو الملف مش موجود بنكمل بالباقي بدل ما نوقف الكل، ونقول إيه اللي ناقص.
    """
    model_dir = get_model_output_path().parent
    models = {}
    for name, filename in MODEL_FILES.items():
        path = model_dir / filename
        if not path.exists():
            print(f"[ناقص] {name}: الملف {path.name} مش موجود. شغّل train الأول.")
            continue
        model = joblib.load(path)
        modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"[تم التحميل] {name:<20} → {type(model).__name__:<22} (آخر تعديل للملف: {modified})")
        models[name] = model
    return models


# ── MANUAL TEST BLOCK ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    models = load_all_models()
    if not models:
        raise SystemExit("مفيش موديلات محفوظة. شغّل: python -m src.models.train")

    # نفس الـ split اللي اتدرّبت عليه الموديلات (نفس RANDOM_STATE وstratify)
    _, X_test, _, y_test = get_evaluation_split()

    print("\n" + "=" * 67)
    print(f"مقارنة الموديلات على بيانات الاختبار ({len(y_test)} عميل، {int(y_test.sum())} سايب)")
    print("=" * 67)
    rows = compare_models(models, X_test, y_test)
    print_comparison(rows)

    report = {"test_size": int(len(y_test)), "churners": int(y_test.sum()), "models": rows}

    # المقارنة العادلة: نفس الـ recall لكل الموديلات، ونشوف مين أدق
    print("=" * 67)
    print("المقارنة عند recall ثابت (الأعدل: بتشيل أثر العتبة)")
    print("=" * 67)
    matched = compare_at_matched_recall(models, X_test, y_test)
    print_matched_recall(matched)
    report["matched_recall"] = matched
    print()

    # تحليل العتبة لـ XGBoost (الموديل الأساسي في المشروع) بنفس المنهج الحدّي
    if "XGBoost" in models:
        print("=" * 67)
        print("تحليل العتبة لـ XGBoost")
        print("=" * 67)
        results = threshold_analysis(models["XGBoost"], X_test, y_test)
        best = find_best_threshold_by_marginal_cost(results)
        print(
            f"\nأرخص خطوة بالتحليل الحدّي: threshold={best['threshold']:.2f} | "
            f"recall={best['recall']:.2%} | precision={best['precision']:.2%} | "
            f"cost={best['cost_per_unit']:.3f}"
        )
        report["xgboost_threshold_analysis"] = [
            {
                "threshold": r["threshold"],
                "recall": float(r["recall"]),
                "precision": float(r["precision"]),
                "f1": float(r["f1"]),
                "missed_churners": int(r["confusion_matrix"][1][0]),
            }
            for r in results
        ]
        report["xgboost_best_by_marginal_cost"] = {
            k: float(v) for k, v in best.items()
        }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nاتحفظت النتايج في: {REPORT_PATH}")