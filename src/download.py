
import time
from pathlib import Path

import requests

from manifest import ROOT, load_manifest

DOCS_DIR = ROOT / "documents"
HEADERS = {"User-Agent": "gcf-rag-assistant/1.0 (portfolio project)"}
TIMEOUT = 60
PAUSE_SECONDS = 1.0  


def download_one(url: str, dest: Path) -> int:
    part = dest.with_suffix(".pdf.part")

    with requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for block in r.iter_content(chunk_size=65536):
                f.write(block)

    with part.open("rb") as f:
        if f.read(4) != b"%PDF":
            part.unlink()
            raise ValueError("server did not return a PDF (HTML error page?)")

    part.rename(dest)
    return dest.stat().st_size


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    manifest = load_manifest()

    got, skipped, failed = 0, 0, []
    for i, row in enumerate(manifest.itertuples(), start=1):
        dest = DOCS_DIR / f"{row.ref}.pdf"

        if dest.exists():
            skipped += 1
            print(f"[{i:>2}/{len(manifest)}] {row.ref}  already have it")
            continue

        try:
            size = download_one(row.pdf_url, dest)
            got += 1
            print(f"[{i:>2}/{len(manifest)}] {row.ref}  {size / 1e6:.1f} MB")
            time.sleep(PAUSE_SECONDS)
        except Exception as exc:
            failed.append((row.ref, str(exc)[:90]))
            print(f"[{i:>2}/{len(manifest)}] {row.ref}  FAILED: {exc}")

    print(f"\ndownloaded {got} | already present {skipped} | failed {len(failed)}")
    for ref, err in failed:
        print(f"  {ref}: {err}")


if __name__ == "__main__":
    main()