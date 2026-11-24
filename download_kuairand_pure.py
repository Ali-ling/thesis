"""Download and safely extract the official KuaiRand-Pure dataset."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import requests


RECORD_API = "https://zenodo.org/api/records/10439422"
TARGET_NAME = "KuaiRand-Pure.tar.gz"
ROOT = Path(__file__).resolve().parent / "data" / "kuairand" / "pure"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe archive path: {member.name}")
        tar.extractall(destination)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    record = requests.get(RECORD_API, timeout=120)
    record.raise_for_status()
    files = record.json().get("files", [])
    selected = next(
        item for item in files if item.get("key") == TARGET_NAME
    )
    target = ROOT / TARGET_NAME
    expected_size = int(selected["size"])
    expected_md5 = selected.get("checksum", "").removeprefix("md5:")
    if not target.exists() or target.stat().st_size != expected_size:
        urls = [
            selected["links"]["self"],
            f"https://zenodo.org/api/records/10439422/files/{TARGET_NAME}?download=1",
            f"https://zenodo.org/records/10439422/files/{TARGET_NAME}?download=1",
            f"https://zenodo.org/record/10439422/files/{TARGET_NAME}",
        ]
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124 Safari/537.36"
            )
        }
        errors: list[str] = []
        for url in urls:
            try:
                with requests.get(
                    url,
                    stream=True,
                    timeout=180,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    with target.open("wb") as file:
                        written = 0
                        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                            if not chunk:
                                continue
                            file.write(chunk)
                            written += len(chunk)
                            if written % (32 * 1024 * 1024) < 4 * 1024 * 1024:
                                print(
                                    f"downloaded {written / 1024**2:.0f} MB",
                                    flush=True,
                                )
                break
            except requests.RequestException as error:
                errors.append(f"{url}: {error}")
        else:
            raise RuntimeError("all KuaiRand download URLs failed: " + " | ".join(errors))
    if target.stat().st_size != expected_size:
        raise RuntimeError(
            f"size mismatch: expected {expected_size}, got {target.stat().st_size}"
        )
    actual_md5 = md5(target)
    if expected_md5 and actual_md5 != expected_md5:
        raise RuntimeError(
            f"MD5 mismatch: expected {expected_md5}, got {actual_md5}"
        )
    extracted = ROOT / "extracted"
    if not extracted.exists():
        safe_extract(target, extracted)
    print(f"archive={target}")
    print(f"size={expected_size}")
    print(f"md5={actual_md5}")
    for path in sorted(extracted.rglob("*")):
        if path.is_file():
            print(f"file={path.relative_to(extracted)} bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
