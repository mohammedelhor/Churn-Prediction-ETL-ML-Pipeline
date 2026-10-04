"""
src/models/tune.py
====================
ضبط hyperparameters لنفس RandomForestClassifier المستخدم في train.py،
باستخدام RandomizedSearchCV بدل GridSearchCV الشامل (بطيء جدًا مع عدد
معايير أكبر من 2-3).

الملف ده بيدرّب على نفس بيانات train.py بالضبط (نفس الاستبعاد لـ
complain، نفس train/test split)، عشان أي مقارنة بين الموديل الأساسي
والمضبوط تبقى عادلة 100%.
"""

import logging
import joblib
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

from src.models.train import (
    load_features,
    split_features_target,
    get_train_test_split,
    get_model_output_path,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)

# نفس قرار train.py — الموديل الرسمي من غير complain (خطر data leakage)
EXCLUDED_COLS = ["complain"]

# مساحة البحث: كل معيار وله مدى منطقي، مش أرقام عشوائية
PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 20),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None],
}


def get_tuned_model_output_path():
    """موديل منفصل عن الأساسي — عشان تقدر تقارن الاتنين، مش تستبدل واحد بالتاني."""
    baseline_path = get_model_output_path()
    return baseline_path.with_name("churn_model_tuned.joblib")


def tune_random_forest(X_train, y_train, n_iter: int = 15, cv: int = 3):
    """
    IN  : بيانات التدريب، عدد المحاولات العشوائية (n_iter)، وعدد
          الطيات لـ cross-validation (cv)
    OUT : (أفضل موديل مدرّب بالكامل، أفضل مجموعة hyperparameters)

    scoring='roc_auc' بدل 'accuracy': فاكر من evaluate.py إن accuracy
    مضللة مع بياناتنا الغير متوازنة (80/20). roc_auc بيقيس قدرة
    الموديل على الترتيب الصح بين الفئتين بغض النظر عن عتبة قرار
    معينة — مناسب هنا لأننا بنضبط العتبة بأنفسنا بعدين على أي حال
    (evaluate.py).

    cv=5: كل تجربة hyperparameters بتتقيّم 5 مرات على تقسيمات مختلفة
    من بيانات التدريب (مش تجربة واحدة بس)، عشان النتيجة تبقى موثوقة
    مش نتيجة حظ من تقسيمة واحدة بالذات.

    الإعدادات الافتراضية (n_iter=15, cv=3 = 45 تجربة فعلية) اتخيّرت
    عشان تاخد وقت معقول على جهاز عادي (دقيقتين تقريبًا). لو عندك وقت
    ومعالج أقوى وعايز بحث أدق، زوّد الرقمين (مثلاً n_iter=30, cv=5)
    عند استدعاء الدالة أو من سطر آخر الملف.
    """
    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    logger.info(f"Best cross-validated ROC-AUC: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_


# ── MANUAL TEST BLOCK ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_features()
    X, y = split_features_target(df)
    X = X.drop(columns=EXCLUDED_COLS, errors="ignore")

    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    logger.info("Starting hyperparameter search...")
    tuned_model, best_params = tune_random_forest(X_train, y_train)

    # مقارنة مباشرة على نفس X_test بالظبط — الموديل الأساسي (المحفوظ
    # من train.py) مقابل الموديل المضبوط، بنفس المقياس (ROC-AUC)
    baseline_model = joblib.load(get_model_output_path())

    baseline_auc = roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1])
    tuned_auc = roc_auc_score(y_test, tuned_model.predict_proba(X_test)[:, 1])

    print("\n" + "=" * 60)
    print("مقارنة: الموديل الأساسي مقابل المضبوط")
    print("=" * 60)
    print(f"Baseline ROC-AUC: {baseline_auc:.4f}")
    print(f"Tuned ROC-AUC:    {tuned_auc:.4f}")
    print(f"الفرق: {tuned_auc - baseline_auc:+.4f}")

    tuned_path = get_tuned_model_output_path()
    joblib.dump(tuned_model, tuned_path)
    logger.info(f"Tuned model saved to: {tuned_path}")