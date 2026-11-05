"""Download and record one immutable model revision in the project cache."""

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from src.translator import MODEL_ID


def main():
    revision = HfApi().model_info(MODEL_ID).sha
    path = snapshot_download(MODEL_ID, revision=revision, cache_dir="models/hub")
    Path("models/selected.json").write_text(json.dumps(
        dict(model=MODEL_ID, revision=revision, path=str(Path(path).resolve())), indent=2
    ) + "\n")
    print(path)


if __name__ == "__main__":
    main()
