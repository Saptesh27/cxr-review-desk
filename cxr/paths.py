from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
NIH_SMALL = DATA / "nih_small"
CHECKPOINTS = ROOT / "checkpoints"
FUSION_PATH = CHECKPOINTS / "fusion.joblib"
DB_PATH = ROOT / "cases.db"
WEB = ROOT / "web"


def ensure_dirs() -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    NIH_SMALL.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
