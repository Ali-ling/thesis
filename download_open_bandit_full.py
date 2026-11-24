"""Download the full BTS/All files from the Hugging Face mirror."""

from __future__ import annotations

import os
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent / "data" / "open_bandit" / "full"
FILES = {
    "bts_all.csv": (
        "https://hf-mirror.com/datasets/zozonext/open-bandit/resolve/main/bts/all/all.csv",
        6_321_017_454,
    ),
    "item_context.csv": (
        "https://hf-mirror.com/datasets/zozonext/open-bandit/resolve/main/bts/all/item_context.csv",
        10_041,
    ),
}


def download(url: str, target: Path, expected_size: int) -> None:
    if target.exists() and target.stat().st_size == expected_size:
        print(f"skip existing {target.name}", flush=True)
        return
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=180) as response:
        response.raise_for_status()
        mode = "ab" if offset and response.status_code == 206 else "wb"
        if mode == "wb":
            offset = 0
        written = offset
        with partial.open(mode) as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024 * 4):
                if not chunk:
                    continue
                file.write(chunk)
                written += len(chunk)
                if written % (256 * 1024 * 1024) < 4 * 1024 * 1024:
                    print(
                        f"{target.name} {written / 1024**2:.0f} MB "
                        f"of {expected_size / 1024**2:.0f} MB",
                        flush=True,
                    )
    actual = partial.stat().st_size
    if actual != expected_size:
        raise RuntimeError(
            f"{target.name}: expected {expected_size} bytes, got {actual}"
        )
    os.replace(partial, target)
    print(f"completed {target.name} bytes={actual}", flush=True)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, (url, size) in FILES.items():
        download(url, ROOT / name, size)


if __name__ == "__main__":
    main()
