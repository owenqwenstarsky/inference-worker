"""
Finds the full GGUF path from the Hugging Face cache using huggingface_hub.
"""

import argparse
import os
from huggingface_hub import snapshot_download, scan_cache_dir


def find_model_path(model_name: str, gguf_in_repo: str) -> str | None:
    """
    Resolve the GGUF file path from the Hugging Face cache.

    Args:
        model_name: Hugging Face repo id (e.g. TheBloke/Mistral-7B-GGUF)
        gguf_in_repo: Relative path to the GGUF file inside the repo

    Returns:
        Full filesystem path to the GGUF file, or None if not found
    """

    # First try the official way: resolve snapshot path from cache
    try:
        snapshot_path = snapshot_download(
            repo_id=model_name,
            local_files_only=True
        )
        candidate = os.path.join(snapshot_path, gguf_in_repo)
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass

    # Fallback: scan cache metadata explicitly (no downloads)
    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id == model_name:
            for revision in repo.revisions:
                candidate = os.path.join(revision.snapshot_path, gguf_in_repo)
                if os.path.isfile(candidate):
                    return candidate

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Find the full GGUF path from the Hugging Face cache."
    )
    parser.add_argument(
        "model",
        type=str,
        help="Hugging Face model repo id (e.g. TheBloke/Mistral-7B-GGUF)",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Relative path to the GGUF file inside the repo",
    )
    args = parser.parse_args()

    model_path = find_model_path(args.model, args.path)

    if model_path is None:
        raise SystemExit("GGUF file not found in Hugging Face cache")

    print(model_path, end="")


if __name__ == "__main__":
    main()
