"""
src/models/cv_threshold.py
==========================
اختيار الـ decision threshold بالـ cross-validation، من غير ما نلمس بيانات الاختبار.

المشكلة اللي بنحلها
-------------------
العتبة 0.24 اخترناها من منحنى precision/recall محسوب على بيانات الاختبار
نفسها، وبعدين قلنا "بنمسك ~75% من السايبين بدقة ~55%" على نفس البيانات.
ده زي إنك تشوف أسئلة الامتحان قبل ما تذاكر: الرقم متفائل شوية. الحل إننا
نختار العتبة من مكان تاني، وبعدين نجرّبها مرة واحدة على الاختبار.

الطريقة (تقدر تعملها بإيدك)
---------------------------
1) ناخد بيانات التدريب بس (X_train) ونقسمها 5 أجزاء (folds) متساوية في نسبة
   الـ churn (StratifiedKFold).
2) لكل جزء: ندرّب الموديل على الأربعة التانيين، ونتنبأ بالجزء اللي ما شافوش.
   بعد الخمسة، كل عميل في التدريب ليه احتمال "out-of-fold" (OOF) — احتمال
   جه من موديل ما شافش العميل ده وقت تدريبه. ده شبه الاختبار الحقيقي.
3) من احتمالات الـ OOF نختار العتبة بقاعدة ثابتة: "أعلى precision بشرط إن
   recall >= الهدف" (نفس قاعدة compare.py، بحيث مفيش تغيير في المنهج).
4) نكرر القاعدة على كل fold لوحده، ونشوف العتبة بتتغير قد إيه بين جزء
   وجزء. لو التغير كبير، يبقى رقم زي 0.24 مش ثابت، ولازم نعرف ده.
5) نطبّق عتبة الـ CV مرة واحدة على الاختبار، ونقارنها بـ 0.24 القديمة.

التشغيل (من جذر المشروع):
    python -m src.models.cv_threshold
"""

import json
import logging

import joblib
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from src.data.paths import PROJECT_ROOT
from src.models.evaluate import get_evaluation_split
from src.models.train import RANDOM_STATE, get_model_output_path, train_model

logger = logging.getLogger(__name__)

# كام % من السايبين عايزين نمسك. 0.75 هو الهدف اللي بنيت عليه قرار 0.24
TARGET_RECALL = 0.75

# أهداف تانية بنطبعها للمقارنة بس
EXTRA_TARGETS = (0.70, 0.80)

N_SPLITS = 5

# العتبة اللي بتشتغل في الـ API دلوقتي (DECISION_THRESHOLD في main.py)
CURRENT_THRESHOLD = 0.24

REPORT_PATH = PROJECT_ROOT / "reports" / "threshold_cv.json"


def pick_threshold_for_recall(y_true, proba, target: float):
    """
    IN  : الحقيقة، الاحتمالات، وهدف الـ recall
    OUT : dict (threshold, recall, precision) أو None لو الهدف مستحيل

    القاعدة: بين كل العتبات اللي recall عندها >= الهدف، ناخد اللي precision
    عندها أعلى (يعني أقل إنذارات كاذبة). نفس منطق compare_at_matched_recall.
    """
    y_true = np.asarray(y_true)
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision وrecall أطول من thresholds بعنصر (نقطة النهاية)، فنقصّهم
    precision, recall = precision[:-1], recall[:-1]
    ok = recall >= target
    if not ok.any():
        return None
    idx = int(precision[ok].argmax())
    return {
        "threshold": float(thresholds[ok][idx]),
        "recall": float(recall[ok][idx]),
        "precision": float(precision[ok][idx]),
    }


def out_of_fold_probabilities(X_train, y_train, n_splits: int = N_SPLITS):
    """
    IN  : بيانات التدريب بس
    OUT : (oof_proba, fold_ids)
          oof_proba: احتمال لكل صف في التدريب، جاي من موديل ما شافش الصف ده
          fold_ids : رقم الـ fold اللي الصف ده كان فيه (عشان نحلل كل جزء لوحده)

    بنستخدم train_model من train.py نفسها (نفس الإعدادات بالظبط)، فالـ CV
    بتقيس الموديل الحقيقي مش نسخة مختلفة.
    shuffle=True مع random_state ثابت: تقسيمة عشوائية لكن قابلة للتكرار.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y_train))
    fold_ids = np.zeros(len(y_train), dtype=int)

    for fold, (fit_idx, hold_idx) in enumerate(skf.split(X_train, y_train)):
        model = train_model(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        oof[hold_idx] = model.predict_proba(X_train.iloc[hold_idx])[:, 1]
        fold_ids[hold_idx] = fold
        logger.info(f"fold {fold + 1}/{n_splits} done")
    return oof, fold_ids


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:
    """recall/precision وعدد الإنذارات الكاذبة عند عتبة معينة."""
    y_true = np.asarray(y_true)
    pred = proba >= threshold
    tp = int((pred & (y_true == 1)).sum())
    fp = int(pred.sum()) - tp
    fn = int(((~pred) & (y_true == 1)).sum())
    return {
        "threshold": float(threshold),
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "flagged": int(pred.sum()),
        "false_alarms": fp,
        "missed_churners": fn,
    }


def run_cv_threshold() -> dict:
    X_train, X_test, y_train, y_test = get_evaluation_split()

    # 1) احتمالات out-of-fold على التدريب
    oof, fold_ids = out_of_fold_probabilities(X_train, y_train)

    # 2) العتبة من الـ OOF كله، لكل هدف
    pooled = {}
    for target in (TARGET_RECALL, *EXTRA_TARGETS):
        pooled[f"{target:.2f}"] = pick_threshold_for_recall(y_train, oof, target)

    # 3) ثبات العتبة: نفس القاعدة على كل fold لوحده (الهدف الأساسي بس)
    per_fold = []
    for fold in np.unique(fold_ids):
        mask = fold_ids == fold
        picked = pick_threshold_for_recall(y_train.to_numpy()[mask], oof[mask], TARGET_RECALL)
        per_fold.append(picked["threshold"])
    per_fold = np.array(per_fold)

    # 4) التطبيق الوحيد على الاختبار: الموديل المحفوظ (مدرّب على كل التدريب)
    #    لو الملف مش موجود ندرّب واحد جديد بنفس الإعدادات.
    model_path = get_model_output_path()
    model = joblib.load(model_path) if model_path.exists() else train_model(X_train, y_train)
    test_proba = model.predict_proba(X_test)[:, 1]

    cv_threshold = pooled[f"{TARGET_RECALL:.2f}"]["threshold"]
    result = {
        "target_recall": TARGET_RECALL,
        "n_splits": N_SPLITS,
        "pooled_oof_choice": pooled,
        "per_fold_thresholds": {
            "values": [float(v) for v in per_fold],
            "mean": float(per_fold.mean()),
            "std": float(per_fold.std(ddof=1)),
            "min": float(per_fold.min()),
            "max": float(per_fold.max()),
        },
        "test_at_cv_threshold": metrics_at_threshold(y_test, test_proba, cv_threshold),
        "test_at_current_threshold": metrics_at_threshold(y_test, test_proba, CURRENT_THRESHOLD),
        "test_positives": int(y_test.sum()),
    }
    return result


def print_result(r: dict) -> None:
    print(f"\n=== العتبة بالـ cross-validation ({r['n_splits']} folds، هدف recall {r['target_recall']:.0%}) ===")

    print("\nالعتبة المختارة من احتمالات OOF على التدريب:")
    for target, p in r["pooled_oof_choice"].items():
        if p is None:
            print(f"  recall {target}: مستحيل")
        else:
            print(f"  recall {target}: threshold={p['threshold']:.3f} "
                  f"(recall {p['recall']:.1%}, precision {p['precision']:.1%})")

    f = r["per_fold_thresholds"]
    print("\nثبات العتبة بين الأجزاء (نفس القاعدة على كل fold):")
    print("  القيم :", ", ".join(f"{v:.3f}" for v in f["values"]))
    print(f"  المتوسط {f['mean']:.3f} | الانحراف المعياري {f['std']:.3f} | من {f['min']:.3f} إلى {f['max']:.3f}")

    print(f"\nعلى الاختبار (الـ {r['test_positives']} سايب) — تطبيق واحد بدون ضبط:")
    print(f"  {'العتبة':<22}{'recall':>9}{'precision':>11}{'flagged':>9}{'false alarms':>14}{'missed':>8}")
    for label, key in (("CV threshold", "test_at_cv_threshold"),
                       ("الحالية (API)", "test_at_current_threshold")):
        m = r[key]
        # نحسب الاسم في متغير لوحده: f-string جوه f-string بنفس نوع الاقتباس
        # مش شغالة في بايثون 3.11 (اللي عندك)، بتشتغل بس من 3.12
        name = label + " (" + format(m["threshold"], ".3f") + ")"
        print(f"  {name:<22}{m['recall']:>9.1%}{m['precision']:>11.1%}"
              f"{m['flagged']:>9}{m['false_alarms']:>14}{m['missed_churners']:>8}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_cv_threshold()
    print_result(result)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\nتم حفظ التقرير في {REPORT_PATH}")