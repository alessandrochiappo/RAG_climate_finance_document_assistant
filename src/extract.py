import json
import re
import unicodedata
from collections import Counter
import pdfplumber

from manifest import ROOT, load_manifest

DOCS_DIR = ROOT / "documents"
PAGES_DIR = ROOT / "data" / "pages"
MIN_PAGE_CHARS = 50
FURNITURE_SHARE = 0.35
FURNITURE_MAX_CHARS = 120

def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text) 
    text = text.replace("\ufffd", "ti")
    text = re.sub(r"[\ue000-\uf8ff]", " ", text)
    text = text.replace("\u00ad", "")           
    text = re.sub(r"[\u2018\u2019]", "'", text) 
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[ \t]+", " ", text)        
    return text.strip()

def find_furniture(pages: list[str]) -> set[str]:
    counts = Counter()
    for text in pages:
        seen = set()
        for line in text.split("\n"):
            line = line.strip()
            if line and len(line) <= FURNITURE_MAX_CHARS:
                seen.add(re.sub(r"\d+", "#", line))
        counts.update(seen)

    threshold = max(3, int(len(pages) * FURNITURE_SHARE))
    return {sig for sig, n in counts.items() if n >= threshold}

def strip_furniture(text: str, furniture: set[str]) -> str:
    kept = [
        line for line in text.split("\n")
        if re.sub(r"\d+", "#", line.strip()) not in furniture
    ]
    return "\n".join(kept).strip()

def extract_document(ref: str) -> list[dict]:
    """One PDF -> a list of page records."""
    pdf_path = DOCS_DIR / f"{ref}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"missing {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        raw = [normalise(page.extract_text() or "") for page in pdf.pages]

    furniture = find_furniture(raw)

    records = []
    for page_no, text in enumerate(raw, start=1):  
        clean = strip_furniture(text, furniture)
        records.append({
            "ref": ref,
            "page": page_no,
            "n_pages": len(raw),
            "text": clean,
            "n_chars": len(clean),
        })
    return records

def main() -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    total_pages = total_chars = 0
    for i, ref in enumerate(manifest["ref"], start=1):
        out = PAGES_DIR / f"{ref}.jsonl"
        if out.exists():
            print(f"[{i:>2}/{len(manifest)}] {ref}  cached")
            continue

        records = extract_document(ref)
        thin = [r["page"] for r in records if r["n_chars"] < MIN_PAGE_CHARS]
        tmp = out.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.rename(out)

        chars = sum(r["n_chars"] for r in records)
        total_pages += len(records)
        total_chars += chars
        print(f"[{i:>2}/{len(manifest)}] {ref}  {len(records):>3} pages  "
              f"{chars:>7,} chars  thin={len(thin)}  {thin[:5]}")

    print(f"\n{total_pages:,} pages | {total_chars:,} characters")

if __name__ == "__main__":
    main()