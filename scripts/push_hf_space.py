"""Create or update the Hugging Face Space from this repo.

Requires a write token in HF_TOKEN or HUGGINGFACE_HUB_TOKEN.
Usage:
    python scripts/push_hf_space.py
    python scripts/push_hf_space.py --repo USER/vesta
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = [
    "app.py",
    "src/vesta",
    "data/cache/bist100.csv",
    "data/cache/usdtry.csv",
    "data/cache/gold.csv",
    "data/vesta_public/events.parquet",
    "data/vesta_public/kap_daily_features.json",
    "data/vesta_public/kap_daily_embeddings_m3.npz",
    "data/vesta_public/kap_inventory.json",
    "data/vesta_public/label_stats.json",
    "results",
    "space/README.md",
    "space/requirements.txt",
]


def _token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "No HF write token. Export HF_TOKEN (write scope) and re-run "
            "python scripts/push_hf_space.py"
        )
    return token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("HF_SPACE_ID", ""))
    args = parser.parse_args()
    token = _token()

    from huggingface_hub import HfApi, create_repo, whoami

    user = whoami(token=token)["name"]
    repo_id = args.repo or f"{user}/vesta"
    api = HfApi(token=token)
    url = create_repo(
        repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        token=token,
    )
    readme = (ROOT / "space" / "README.md").read_text(encoding="utf-8")
    reqs = (ROOT / "space" / "requirements.txt").read_text(encoding="utf-8")
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )
    api.upload_file(
        path_or_fileobj=reqs.encode("utf-8"),
        path_in_repo="requirements.txt",
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )
    for rel in INCLUDE:
        if rel.startswith("space/"):
            continue
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"missing {rel}")
        if src.is_dir():
            api.upload_folder(
                folder_path=str(src),
                path_in_repo=rel,
                repo_id=repo_id,
                repo_type="space",
                token=token,
                ignore_patterns=["__pycache__", "*.pyc", "static/.DS_Store"],
            )
        else:
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=rel,
                repo_id=repo_id,
                repo_type="space",
                token=token,
            )
    print(f"Space: https://huggingface.co/spaces/{repo_id}")
    print(url)


if __name__ == "__main__":
    sys.exit(main() or 0)
