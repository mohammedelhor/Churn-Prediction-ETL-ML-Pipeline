"""
src/data/paths.py
==================
المكان المشترك الوحيد لحساب PROJECT_ROOT وقراءة config.yaml.

ليه عملنا الملف ده؟
كان نفس الكود (PROJECT_ROOT, CONFIG_PATH, load_config) مكرر حرفيًا في
extract.py و load.py. أي تعديل (زي تغيير مكان config.yaml) كان محتاج
يتعمل في مكانين. القاعدة: أي منطق بيتكرر في أكتر من ملف، انقله لمكان
واحد والباقي يستورد منه.
"""

from pathlib import Path
import yaml

# paths.py موجود في: <ROOT>/src/data/paths.py — نفس مكان extract.py
# بالظبط، فنفس الحسبة (.parents[2]) بتنطبق هنا كمان
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """قراءة config.yaml وترجيعه كـ dict عادي."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)