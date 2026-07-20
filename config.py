"""
ZTF_prompt config
Environment variables (.env): LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, ZTF_DATA_DIR
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# API
API_KEY = os.environ.get("LLM_API_KEY", "")
API_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.llm.ustc.edu.cn/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3.6-reasoner")
TEMPERATURE = 0.0

if not API_KEY:
    raise RuntimeError("Set LLM_API_KEY in ZTF_prompt/.env")

# Paths
PROJECT_ROOT = Path(__file__).parent
SOURCES_DIR = PROJECT_ROOT / "sources"
INDEX_FILE = PROJECT_ROOT / "index.json"
RESULTS_DIR = PROJECT_ROOT / "results"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# ZTF raw data
ZTF_DATA_DIR = Path(
    os.environ.get(
        "ZTF_DATA_DIR",
        str(Path.home() / "AppData" / "VScode" / "TDeck" / "ZTF_TDE" / "data"),
    )
)
ZTF_FLUX_DIR = ZTF_DATA_DIR / "TS" / "Flux"
ZTF_EARLY_DIR = ZTF_DATA_DIR / "TS"

# Classification
CLASSES = ["TDE", "SN"]#, "Others", "AGN"]
N_SHOT_TEXT = 3
N_SHOT_MULTIMODAL = 1

# Feature computation
MAX_LEN = 100
MIN_PTS = 5
BAND_MAP = {1: "g", 2: "r", 3: "u"}

# Ensure dirs exist
for d in [SOURCES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
