"""
src/models/evaluate.py
========================
تقييم الموديل المحفوظ (النسخة الواقعية، من غير complain) بمقاييس
مناسبة لمشكلة غير متوازنة الفئات — accuracy وحدها مضللة هنا (شرحنا ليه).
"""

import logging
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

from src.models.train import (
    load_features,
    split_features_target,
    get_train_test_split,
    get_model_output_path,
)

logger = logging.getLogger(__name__)

# لازم نفس الاستبعاد بالظبط اللي حصل وقت تدريب الموديل المحفوظ،
# وإلا عدد الأعمدة اللي بنبعتها للموديل هيختلف عن اللي اتدرب عليه
EXCLUDED_COLS = ["complain"]


def load_trained_model():
    return joblib.load(get_model_output_path())


def get_evaluation_split():
    """
    نفس RANDOM_STATE وTEST_SIZE بالظبط من train.py (لأننا بنستورد
    get_train_test_split منه مباشرة، مش بنعيد كتابتها) — يضمن إن
    X_test هنا هو نفسه X_test اللي الموديل ما شافوش وقت التدريب.
    """
    df = load_features()
    X, y = split_features_target(df)
    X = X.drop(columns=EXCLUDED_COLS, errors="ignore")
    return get_train_test_split(X, y)


def evaluate_model(model, X_test, y_test):
    """
    IN  : الموديل المدرّب، وبيانات الاختبار (X_test, y_test)
    OUT : dict فيه التقرير الكامل، confusion matrix، وROC-AUC

    predict() بيرجع قرار نهائي (0 أو 1). predict_proba()[:, 1] بيرجع
    "احتمال" إن العميل هيسيب (رقم بين 0 و1) — محتاجينه لـ ROC-AUC
    اللي بيقيس جودة الترتيب (ranking) مش القرار النهائي بس.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["Stayed", "Churned"])
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"Confusion Matrix (rows=actual, cols=predicted):\n{cm}")
    logger.info(f"ROC-AUC: {auc:.4f}")

    return {"report": report, "confusion_matrix": cm, "roc_auc": auc}


def evaluate_at_threshold(model, X_test, y_test, threshold: float):
    """
    IN  : الموديل، بيانات الاختبار، والعتبة الجديدة (بين 0 و1)
    OUT : dict فيه recall/precision/f1 للفئة "سايب" تحديدًا، عند
          العتبة دي بالذات

    predict() الافتراضية بتستخدم عتبة 0.5 مدفونة جواها من غير ما
    تقدر تغيّرها. الحل: ناخد الاحتمالات الخام (predict_proba) ونعمل
    المقارنة بأنفسنا بالعتبة اللي اخترناها.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_proba >= threshold).astype(int)

    report = classification_report(
        y_test, y_pred_custom, target_names=["Stayed", "Churned"], output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred_custom)

    churned = report["Churned"]
    return {
        "threshold": threshold,
        "recall": churned["recall"],
        "precision": churned["precision"],
        "f1": churned["f1-score"],
        "confusion_matrix": cm,
    }


def threshold_analysis(model, X_test, y_test, thresholds=None):
    """
    بيطبع جدول مقارنة بين كذا عتبة مختلفة، وبيرجّع النتايج كـ list
    (مش بس بيطبعها) عشان دالة تانية (find_best_threshold_by_marginal_cost)
    تقدر تستخدمها في حساب لاحق، بدل ما تعيد حساب كل شيء من الصفر.

    كل ما العتبة تقل، recall يزيد (نمسك حالات أكتر) لكن precision يقل
    (نغلط في حالات أكتر) — العكس صحيح كمان.
    """
    if thresholds is None:
        thresholds = [0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]

    print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>8}   Missed churners")
    print("-" * 60)

    results = []
    for t in thresholds:
        result = evaluate_at_threshold(model, X_test, y_test, t)
        missed = result["confusion_matrix"][1][0]  # فعليًا سايب، اتوقع فضل
        print(
            f"{t:>10.2f} {result['recall']:>8.2%} {result['precision']:>10.2%} "
            f"{result['f1']:>8.2%}   {missed}"
        )
        results.append(result)

    return results


def find_best_threshold_by_marginal_cost(results: list) -> dict:
    """
    IN  : list النتايج من threshold_analysis — لازم تكون مرتبة من
          العتبة الأكبر للأصغر (زي الترتيب الافتراضي بالظبط)
    OUT : dict فيه أفضل threshold، وrecall/precision بتاعها، وتكلفة
          الوحدة اللي خلتها تفوز

    نفس المنهج اللي عملناه يدوي بالضبط:
    لكل زوج عتبات متتاليين (الأكبر ثم الأصغر بعدها)، نحسب:
        كسب Recall     = recall(الأصغر) - recall(الأكبر)
        خسارة Precision = precision(الأكبر) - precision(الأصغر)
        تكلفة الوحدة    = خسارة Precision ÷ كسب Recall

    الخطوة صاحبة أقل تكلفة وحدة = "أرخص صفقة" في الجدول كله.
    العتبة اللي بترجعها الدالة هي العتبة الأصغر بتاع الخطوة الفايزة —
    يعني "الوجهة" اللي وصلنا لها بأرخص تكلفة، مش نقطة الانطلاق.
    """
    if len(results) < 2:
        raise ValueError("محتاج نتايج عتبتين على الأقل للمقارنة")

    best_result = None
    best_cost = float("inf")

    for i in range(len(results) - 1):
        higher = results[i]      # العتبة الأكبر
        lower = results[i + 1]   # العتبة الأصغر (التالية في الترتيب)

        gain_recall = lower["recall"] - higher["recall"]
        loss_precision = higher["precision"] - lower["precision"]

        if gain_recall <= 0:
            continue  # تجاهل أي خطوة معكوسة (نادر، لكن ممكن رقميًا)

        cost_per_unit = loss_precision / gain_recall

        if cost_per_unit < best_cost:
            best_cost = cost_per_unit
            best_result = lower

    return {
        "threshold": best_result["threshold"],
        "recall": best_result["recall"],
        "precision": best_result["precision"],
        "cost_per_unit": best_cost,
    }


# ── MANUAL TEST BLOCK ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = load_trained_model()
    X_train, X_test, y_train, y_test = get_evaluation_split()

    evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("تحليل العتبة (Threshold Analysis)")
    print("=" * 60)
    results = threshold_analysis(model, X_test, y_test)

    best = find_best_threshold_by_marginal_cost(results)
    print("\n" + "=" * 60)
    print("أفضل عتبة بناءً على التحليل الحدّي (Marginal Analysis)")
    print("=" * 60)
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Recall: {best['recall']:.2%} | Precision: {best['precision']:.2%}")
    print(f"تكلفة الوحدة: {best['cost_per_unit']:.3f} (أقل تكلفة في الجدول)")