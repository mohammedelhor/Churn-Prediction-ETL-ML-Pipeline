import logging
# from os import path
import pandas as pd
from pathlib import Path
import yaml

# logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

def load_config(config_path = CONFIG_PATH) -> dict:
    """ read the config file and return the config as a dictionary"""
    with open (config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract(path: Path | None = None) -> pd.DataFrame:

    """ لو معملتش تمرير ل config.yaml ال path يبني المسار تلقايا
    """

    if path is None :
        config = load_config()
        path = PROJECT_ROOT / config["data"]["raw_dir"] / config["data"]["raw_filename"]

    logger.info(f"Extracting data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Extracted shape: {df.shape}")
    return df

# الجزء الخاص باستدعاء الداله

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = extract()
    print(df.head())
    print(f"Shape: {df.shape}")


     