"""Small shared helpers: locating the project root, loading config, resolving paths.

You normally do not run this file directly. The other scripts import from it.
"""
from pathlib import Path
import yaml

# The project root is the folder that contains this `src` directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read config/config.yaml into a Python dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Turn a path from the config (relative to the project root) into a full path."""
    return REPO_ROOT / relative_path
