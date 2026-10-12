"""
src/api/schemas.py
====================
Pydantic models — بتتحقق من شكل البيانات الداخلة والخارجة تلقائيًا.
لو عميل بعت age = "cat" مثلاً، FastAPI هيرفضه فورًا برسالة واضحة،
من غير ما نكتب أي فحص يدوي بنفسنا.
"""

from typing import Literal

from pydantic import BaseModel, Field

# القيم المسموحة لكل عمود categorical: نفس ALLOWED_CATEGORIES في validate.py.
# ليه Literal؟ لو وصل "Italy" مثلًا، prepare_single_customer كان هيعتبره
# الفئة الأساسية (France) من غير أي تحذير، والموديل يتنبأ على بيانات
# غلط. بـ Literal، FastAPI بيرفض الطلب فورًا برسالة 422 واضحة.
Geography = Literal["France", "Germany", "Spain"]
Gender = Literal["Male", "Female"]
CardType = Literal["DIAMOND", "GOLD", "SILVER", "PLATINUM"]


class CustomerInput(BaseModel):
    """
    نفس أعمدة العميل الخام (زي ما هي في transform.py)، من غير:
    - customer_id, surname: معرّفات، مش سمات تنبؤية
    - complain: مستبعد من الموديل النهائي (خطر data leakage موثّق
      في build_features.py) — وأصلاً القيمة دي مش متاحة منطقيًا
      وقت التنبؤ الحقيقي (العميل لسه ما قرّرش يسيب أو يشتكي)
    """
    credit_score: int = Field(..., ge=300, le=850, description="نطاق فعلي من EDA: 350-850")
    geography: Geography = Field(..., description="France, Germany, أو Spain")
    gender: Gender = Field(..., description="Male أو Female")
    age: int = Field(..., ge=18, le=100)
    tenure: int = Field(..., ge=0, le=10)
    balance: float = Field(..., ge=0)
    num_of_products: int = Field(..., ge=1, le=4)
    has_cr_card: int = Field(..., ge=0, le=1)
    is_active_member: int = Field(..., ge=0, le=1)
    estimated_salary: float = Field(..., ge=0)
    satisfaction_score: int = Field(..., ge=1, le=5)
    card_type: CardType = Field(..., description="DIAMOND, GOLD, SILVER, أو PLATINUM")
    point_earned: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "credit_score": 619,
                "geography": "France",
                "gender": "Female",
                "age": 42,
                "tenure": 2,
                "balance": 0.0,
                "num_of_products": 1,
                "has_cr_card": 1,
                "is_active_member": 1,
                "estimated_salary": 101348.88,
                "satisfaction_score": 2,
                "card_type": "DIAMOND",
                "point_earned": 464,
            }
        }


class PredictionOutput(BaseModel):
    churn_probability: float = Field(..., description="احتمالية المغادرة، بين 0 و1")
    will_churn: bool = Field(..., description="القرار النهائي بناءً على العتبة المعتمدة (DECISION_THRESHOLD في main.py)")
    threshold_used: float = Field(..., description=" العتبه اللي اتأخذ عليها القرار النهائي (DECISION_THRESHOLD في main.py)") 