"""Download the official 10k-row Open Bandit sample for data auditing."""

from __future__ import annotations

from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent / "data" / "open_bandit" / "sample"
FILES = {
    "bts_all.csv": [
        "https://cdn.jsdelivr.net/gh/st-tech/zr-obp@master/obd/bts/all/all.csv",
        "https://github.com/st-tech/zr-obp/raw/refs/heads/master/obd/bts/all/all.csv",
        "https://raw.githubusercontent.com/st-tech/zr-obp/master/obd/bts/all/all.csv",
    ],
    "random_all.csv": [
        "https://cdn.jsdelivr.net/gh/st-tech/zr-obp@master/obd/random/all/all.csv",
        "https://github.com/st-tech/zr-obp/raw/refs/heads/master/obd/random/all/all.csv",
        "https://raw.githubusercontent.com/st-tech/zr-obp/master/obd/random/all/all.csv",
    ],
    "item_context.csv": [
        "https://cdn.jsdelivr.net/gh/st-tech/zr-obp@master/obd/bts/all/item_context.csv",
        "https://github.com/st-tech/zr-obp/raw/refs/heads/master/obd/bts/all/item_context.csv",
        "https://raw.githubusercontent.com/st-tech/zr-obp/master/obd/bts/all/item_context.csv",
    ],
}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, urls in FILES.items():
        target = ROOT / name
        errors: list[str] = []
        for url in urls:
            try:
                with requests.get(url, stream=True, timeout=120) as response:
                    response.raise_for_status()
                    with target.open("wb") as file:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                file.write(chunk)
                break
            except requests.RequestException as error:
                errors.append(f"{url}: {error}")
        else:
            raise RuntimeError(f"all download sources failed for {name}: {errors}")
        print(f"{name} bytes={target.stat().st_size}")


if __name__ == "__main__":
    main()
