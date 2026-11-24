"""List file sizes for the official Hugging Face Open Bandit mirror."""

from __future__ import annotations

import requests


URLS = [
    (
        "https://huggingface.co/api/datasets/zozonext/open-bandit/tree/main"
        "?recursive=true&expand=true"
    ),
    (
        "https://hf-mirror.com/api/datasets/zozonext/open-bandit/tree/main"
        "?recursive=true&expand=true"
    ),
]


def main() -> None:
    payload = None
    errors: list[str] = []
    for url in URLS:
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as error:
            errors.append(f"{url}: {error}")
    if payload is None:
        raise RuntimeError("all mirror requests failed: " + " | ".join(errors))
    rows = []
    for item in payload:
        if item.get("type") != "file":
            continue
        size = item.get("size")
        if isinstance(size, dict):
            size = size.get("regular")
        rows.append((item["path"], int(size or 0)))
    for path, size in sorted(rows):
        print(f"{size:>14} {size / 1024**2:>10.1f} MB {path}")


if __name__ == "__main__":
    main()
