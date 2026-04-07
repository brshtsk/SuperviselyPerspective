import os
from pathlib import Path

import requests


DEFAULT_MODEL_URL = "https://huggingface.co/mitbersh/car-view/resolve/main/car_view_model.pth"
DEFAULT_MODEL_NAME = "car_view_model.pth"


def get_default_cache_dir() -> Path:
    cache_dir = os.getenv("CAR_VIEW_MODEL_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)

    base_dir = Path(__file__).resolve().parents[1]
    return base_dir / ".cache"


def ensure_model_downloaded(model_url: str = DEFAULT_MODEL_URL, model_name: str = DEFAULT_MODEL_NAME) -> str:
    cache_dir = get_default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / model_name
    if model_path.exists() and model_path.stat().st_size > 0:
        return str(model_path)

    tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    with requests.get(model_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)

    os.replace(tmp_path, model_path)
    return str(model_path)

