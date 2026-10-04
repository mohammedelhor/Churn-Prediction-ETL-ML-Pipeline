# """
# src/models/train.py
# ====================
# تدريب أول موديل — Random Forest، بناءً على قرارين اتاخدوا من نتايج
# EDA فعلية:
# 1) Random Forest مش Logistic Regression: علاقة num_of_products
#    بالـ churn غير خطية (شكل U مقلوب)، وRandom Forest بيمسكها من غير
#    أي معالجة إضافية
# 2) class_weight='balanced': هدف المشروع اكتشاف "مين هيسيب" (الفئة
#    الأقل، ~20% من العملاء)، مش مجرد accuracy عالي على حساب تجاهلهم
# """

# import logging
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
# import pandas as pd

# from src.data.paths import PROJECT_ROOT, load_config

# logger = logging.getLogger(__name__)

# TARGET_COL = "exited"

# # ثابت عشان أي تشغيلة تانية (بما فيها evaluate.py بعدين) تاخد
# # بالظبط نفس تقسيمة train/test — لازم يكون نفس الرقم في الاتنين
# RANDOM_STATE = 42
# TEST_SIZE = 0.2


# def get_features_path():
#     config = load_config()
#     return PROJECT_ROOT / config["data"]["processed_dir"] / "customer_churn_features.parquet"


# def get_model_output_path():
#     """
#     مجلد joblib/ موجود بالفعل في هيكل مشروعك (شايفينه في الصورة الأولى)
#     — هنا مكان حفظ الموديلات المدرّبة تقليديًا في مشاريع الـ ML.
#     """
#     config = load_config()
#     model_dir = PROJECT_ROOT / "joblib"
#     model_dir.mkdir(parents=True, exist_ok=True)
#     return model_dir / "churn_model.joblib"


# def load_features() -> pd.DataFrame:
#     return pd.read_parquet(get_features_path())


# def split_features_target(df: pd.DataFrame):
#     """
#     IN  : df كامل (features + target exited)
#     OUT : X (كل الأعمدة عدا exited), y (عمود exited بس)
#     """
#     X = df.drop(columns=[TARGET_COL])
#     y = df[TARGET_COL]
#     return X, y


# def get_train_test_split(X: pd.DataFrame, y: pd.Series):
#     """
#     stratify=y مهم جدًا هنا — بيضمن إن نسبة الـ churn (~20%) تفضل
#     ثابتة تقريبًا في train وtest مع بعض. من غيرها، ممكن بالصدفة
#     يقع تقسيم فيه كل حالات الـ churn في جانب واحد بس، وده يبوّظ
#     التقييم بالكامل.

#     دالة منفصلة (مش جوه main) عشان evaluate.py بعدين يقدر يستوردها
#     مباشرة ويضمن بالضبط نفس تقسيمة train/test، مش يعيد كتابتها.
#     """
#     return train_test_split(
#         X, y,
#         test_size=TEST_SIZE,
#         random_state=RANDOM_STATE,
#         stratify=y,
#     )


# def train_model(X_train, y_train) -> RandomForestClassifier:
#     model = RandomForestClassifier(
#         n_estimators=200,
#         class_weight="balanced",
#         random_state=RANDOM_STATE,
#         n_jobs=-1,   # يستخدم كل أنوية المعالج المتاحة، تدريب أسرع
#     )
#     model.fit(X_train, y_train)
#     return model


# def compute_scale_pos_weight(y_train) -> float:
#     """
#     IN  : y_train (عمود الفئة الحقيقي)
#     OUT : نسبة "الفضلوا" لـ "السابوا" — محسوبة من البيانات الفعلية،
#           مش رقم مكتوب يدويًا (زي الملف اللي جبته وكان حاطط 3 ثابتة
#           من غير حساب). XGBoost مالوش class_weight='balanced' زي
#           sklearn، البديل بتاعه هو scale_pos_weight بنفس الفكرة.
#     """
#     negative = (y_train == 0).sum()
#     positive = (y_train == 1).sum()
#     return negative / positive


# def train_xgboost_model(X_train, y_train) -> XGBClassifier:
#     """
#     ملاحظة توافق: use_label_encoder اتشالت نهائيًا من XGBoost بداية
#     من إصدار 1.6 — لو حطيتها هنا، هترمي TypeError فورًا. مش موجودة
#     عمدًا هنا، مش نسيان.
#     """
#     scale_pos_weight = compute_scale_pos_weight(y_train)
#     logger.info(f"scale_pos_weight (محسوب من البيانات): {scale_pos_weight:.2f}")

#     model = XGBClassifier(
#         n_estimators=200,
#         max_depth=6,
#         learning_rate=0.1,
#         scale_pos_weight=scale_pos_weight,
#         random_state=RANDOM_STATE,
#         eval_metric="logloss",
#         n_jobs=-1,
#     )
#     model.fit(X_train, y_train)
#     return model


# def train_and_evaluate_variant(df: pd.DataFrame, exclude_cols: list, label: str):
#     """
#     IN  : df كامل الـ features، قايمة أعمدة نستبعدها (ممكن تبقى فاضية)،
#           واسم وصفي للنسخة (للطباعة بس)
#     OUT : dict فيه الموديل، الدقة، وأهم 5 features — عشان نقارن
#           نسختين مع بعض بسهولة من غير تكرار كود التدريب كامل مرتين

#     ده أسلوب "parametrize بدل تكرار": بدل ما نكتب train_model مرتين
#     بفرق سطر واحد بينهم، بنمرر الاختلاف (exclude_cols) كمدخل للدالة
#     """
#     X, y = split_features_target(df)
#     X = X.drop(columns=exclude_cols, errors="ignore")

#     X_train, X_test, y_train, y_test = get_train_test_split(X, y)
#     model = train_model(X_train, y_train)

#     train_acc = model.score(X_train, y_train)
#     test_acc = model.score(X_test, y_test)

#     importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

#     logger.info(f"\n=== {label} ===")
#     logger.info(f"Features used: {list(X.columns)}")
#     logger.info(f"Train accuracy: {train_acc:.4f} | Test accuracy: {test_acc:.4f}")
#     logger.info(f"Top 5 features:\n{importances.head(5)}")

#     return {
#         "label": label,
#         "model": model,
#         "train_acc": train_acc,
#         "test_acc": test_acc,
#         "importances": importances,
#     }


# # ── MANUAL TEST BLOCK ────────────────────────────────────────────────────
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     df = load_features()

#     # النسختين من قبل: مع complain (للتوثيق بس) ومن غيره (الواقعية)
#     with_complain = train_and_evaluate_variant(df, exclude_cols=[], label="RF WITH complain")
#     without_complain = train_and_evaluate_variant(
#         df, exclude_cols=["complain"], label="RF WITHOUT complain"
#     )

#     # النسخة الجديدة: XGBoost على نفس الـ features الواقعية (من غير complain)
#     X, y = split_features_target(df)
#     X = X.drop(columns=["complain"], errors="ignore")
#     X_train, X_test, y_train, y_test = get_train_test_split(X, y)

#     xgb_model = train_xgboost_model(X_train, y_train)
#     xgb_test_acc = xgb_model.score(X_test, y_test)
#     logger.info(f"\n=== XGBoost WITHOUT complain ===")
#     logger.info(f"Test accuracy: {xgb_test_acc:.4f}")

#     print("\n" + "=" * 60)
#     print("مقارنة نهائية (من غير complain في الاتنين)")
#     print("=" * 60)
#     print(f"Random Forest: test accuracy = {without_complain['test_acc']:.4f}")
#     print(f"XGBoost:       test accuracy = {xgb_test_acc:.4f}")

#     # نحفظ XGBoost كموديل رسمي جديد للمشروع — منفصل عن RF عشان
#     # تقدر ترجع تقارن بينهم لاحقًا بـ evaluate.py
#     model_path = get_model_output_path()
#     xgb_model_path = model_path.with_name("churn_model_xgboost.joblib")
#     joblib.dump(xgb_model, xgb_model_path)
#     logger.info(f"\nXGBoost model saved: {xgb_model_path}")
"""
src/models/train.py
====================
تدريب أول موديل — Random Forest، بناءً على قرارين اتاخدوا من نتايج
EDA فعلية:
1) Random Forest مش Logistic Regression: علاقة num_of_products
   بالـ churn غير خطية (شكل U مقلوب)، وRandom Forest بيمسكها من غير
   أي معالجة إضافية
2) class_weight='balanced': هدف المشروع اكتشاف "مين هيسيب" (الفئة
   الأقل، ~20% من العملاء)، مش مجرد accuracy عالي على حساب تجاهلهم
"""

"""
src/models/train.py
====================
تدريب أول موديل — Random Forest، بناءً على قرارين اتاخدوا من نتايج
EDA فعلية:
1) Random Forest مش Logistic Regression: علاقة num_of_products
   بالـ churn غير خطية (شكل U مقلوب)، وRandom Forest بيمسكها من غير
   أي معالجة إضافية
2) class_weight='balanced': هدف المشروع اكتشاف "مين هيسيب" (الفئة
   الأقل، ~20% من العملاء)، مش مجرد accuracy عالي على حساب تجاهلهم
"""

import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd

from src.data.paths import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

TARGET_COL = "exited"

# ثابت عشان أي تشغيلة تانية (بما فيها evaluate.py بعدين) تاخد
# بالظبط نفس تقسيمة train/test — لازم يكون نفس الرقم في الاتنين
RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_features_path():
    config = load_config()
    return PROJECT_ROOT / config["data"]["processed_dir"] / "customer_churn_features.parquet"


def get_model_output_path():
    """
    مجلد joblib/ موجود بالفعل في هيكل مشروعك (شايفينه في الصورة الأولى)
    — هنا مكان حفظ الموديلات المدرّبة تقليديًا في مشاريع الـ ML.
    """
    config = load_config()
    model_dir = PROJECT_ROOT / "joblib"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "churn_model.joblib"


def load_features() -> pd.DataFrame:
    return pd.read_parquet(get_features_path())


def split_features_target(df: pd.DataFrame):
    """
    IN  : df كامل (features + target exited)
    OUT : X (كل الأعمدة عدا exited), y (عمود exited بس)
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def get_train_test_split(X: pd.DataFrame, y: pd.Series):
    """
    stratify=y مهم جدًا هنا — بيضمن إن نسبة الـ churn (~20%) تفضل
    ثابتة تقريبًا في train وtest مع بعض. من غيرها، ممكن بالصدفة
    يقع تقسيم فيه كل حالات الـ churn في جانب واحد بس، وده يبوّظ
    التقييم بالكامل.

    دالة منفصلة (مش جوه main) عشان evaluate.py بعدين يقدر يستوردها
    مباشرة ويضمن بالضبط نفس تقسيمة train/test، مش يعيد كتابتها.
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_model(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,   # يستخدم كل أنوية المعالج المتاحة، تدريب أسرع
    )
    model.fit(X_train, y_train)
    return model


def compute_scale_pos_weight(y_train) -> float:
    """
    IN  : y_train (عمود الفئة الحقيقي)
    OUT : نسبة "الفضلوا" لـ "السابوا" — محسوبة من البيانات الفعلية،
          مش رقم مكتوب يدويًا (زي الملف اللي جبته وكان حاطط 3 ثابتة
          من غير حساب). XGBoost مالوش class_weight='balanced' زي
          sklearn، البديل بتاعه هو scale_pos_weight بنفس الفكرة.
    """
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    return negative / positive


def train_xgboost_model(X_train, y_train) -> XGBClassifier:
    """
    ملاحظة توافق: use_label_encoder اتشالت نهائيًا من XGBoost بداية
    من إصدار 1.6 — لو حطيتها هنا، هترمي TypeError فورًا. مش موجودة
    عمدًا هنا، مش نسيان.
    """
    scale_pos_weight = compute_scale_pos_weight(y_train)
    logger.info(f"scale_pos_weight (محسوب من البيانات): {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train) -> Pipeline:
    """
    Pipeline بيضم StandardScaler + LogisticRegression مع بعض —
    مهم جدًا هنا: Logistic Regression حساس لمقياس الأعمدة (balance
    بالآلاف مقابل tenure من 0-10)، عكس RF وXGBoost اللي مش بيهمهم
    فرق المقياس ده خالص. من غير scaling، أي فشل هيبقى حكم مش عادل
    على الخوارزمية نفسها.

    Pipeline بيضمن كمان إن نفس الـ StandardScaler المتدرّب على
    X_train بالظبط هو اللي هيتطبق على أي بيانات تانية بعدين (X_test،
    أو بيانات حقيقية وقت التنبؤ) — من غير ما تسرّب معلومات من الاختبار
    للتدريب بالغلط (لو عملت fit_transform على الكل مع بعض).
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000,
        )),
    ])
    model.fit(X_train, y_train)
    return model


def train_and_evaluate_variant(df: pd.DataFrame, exclude_cols: list, label: str):
    """
    IN  : df كامل الـ features، قايمة أعمدة نستبعدها (ممكن تبقى فاضية)،
          واسم وصفي للنسخة (للطباعة بس)
    OUT : dict فيه الموديل، الدقة، وأهم 5 features — عشان نقارن
          نسختين مع بعض بسهولة من غير تكرار كود التدريب كامل مرتين

    ده أسلوب "parametrize بدل تكرار": بدل ما نكتب train_model مرتين
    بفرق سطر واحد بينهم، بنمرر الاختلاف (exclude_cols) كمدخل للدالة
    """
    X, y = split_features_target(df)
    X = X.drop(columns=exclude_cols, errors="ignore")

    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    model = train_model(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    logger.info(f"\n=== {label} ===")
    logger.info(f"Features used: {list(X.columns)}")
    logger.info(f"Train accuracy: {train_acc:.4f} | Test accuracy: {test_acc:.4f}")
    logger.info(f"Top 5 features:\n{importances.head(5)}")

    return {
        "label": label,
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "importances": importances,
    }


# ── MANUAL TEST BLOCK ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_features()

    # النسختين من قبل: مع complain (للتوثيق بس) ومن غيره (الواقعية)
    with_complain = train_and_evaluate_variant(df, exclude_cols=[], label="RF WITH complain")
    without_complain = train_and_evaluate_variant(
        df, exclude_cols=["complain"], label="RF WITHOUT complain"
    )

    # النسخة الجديدة: XGBoost على نفس الـ features الواقعية (من غير complain)
    X, y = split_features_target(df)
    X = X.drop(columns=["complain"], errors="ignore")
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    xgb_model = train_xgboost_model(X_train, y_train)
    xgb_test_acc = xgb_model.score(X_test, y_test)
    logger.info(f"\n=== XGBoost WITHOUT complain ===")
    logger.info(f"Test accuracy: {xgb_test_acc:.4f}")

    logreg_model = train_logistic_regression(X_train, y_train)
    logreg_test_acc = logreg_model.score(X_test, y_test)
    logger.info(f"\n=== Logistic Regression WITHOUT complain ===")
    logger.info(f"Test accuracy: {logreg_test_acc:.4f}")

    print("\n" + "=" * 60)
    print("مقارنة نهائية (من غير complain في الكل)")
    print("=" * 60)
    print(f"Random Forest:       test accuracy = {without_complain['test_acc']:.4f}")
    print(f"XGBoost:             test accuracy = {xgb_test_acc:.4f}")
    print(f"Logistic Regression: test accuracy = {logreg_test_acc:.4f}")

    # نحفظ الثلاثة كلهم — عشان نقدر نرجع نقارنهم بـ evaluate.py
    # بمقاييس أعمق (Recall/ROC-AUC)، مش accuracy بس
    model_path = get_model_output_path()
    xgb_model_path = model_path.with_name("churn_model_xgboost.joblib")
    logreg_model_path = model_path.with_name("churn_model_logreg.joblib")

    joblib.dump(xgb_model, xgb_model_path)
    joblib.dump(logreg_model, logreg_model_path)
    logger.info(f"\nXGBoost model saved: {xgb_model_path}")
    logger.info(f"Logistic Regression model saved: {logreg_model_path}")