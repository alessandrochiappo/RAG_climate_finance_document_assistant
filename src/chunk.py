import json
import re

from manifest import ROOT, load_manifest

PAGES_DIR = ROOT / "data" / "pages"
OUT_PATH = ROOT / "data" / "chunks.jsonl"

TARGET_CHARS = 800     
OVERLAP_CHARS = 150   
MIN_CHUNK_CHARS = 100  
MIN_PAGE_CHARS = 50   
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def split_page(text: str) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " ", text).strip()
    sentences = SENTENCE_BREAK.split(flat)
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) > TARGET_CHARS:
            units.extend(hard_split(sentence, TARGET_CHARS))
        else:
            units.append(sentence)

    chunks: list[str] = []
    current = ""

    for unit in units:
        if current and len(current) + 1 + len(unit) > TARGET_CHARS:
            chunks.append(current)
            tail = current[-OVERLAP_CHARS:]
            space = tail.find(" ")
            current = (tail[space + 1:] if space != -1 else tail) + " " + unit
        else:
            current = f"{current} {unit}".strip()

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]

def hard_split(text: str, size: int) -> list[str]:
    pieces, current = [], ""
    for word in text.split(" "):
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces

def main() -> None:
    manifest = load_manifest()
    meta_by_ref = manifest.set_index("ref").to_dict("index")

    n_chunks = n_pages_used = n_pages_skipped = 0
    sizes: list[int] = []

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for ref in manifest["ref"]:
            meta = meta_by_ref[ref]
            per_doc = 0

            for line in (PAGES_DIR / f"{ref}.jsonl").open(encoding="utf-8"):
                page = json.loads(line)

                if page["n_chars"] < MIN_PAGE_CHARS:
                    n_pages_skipped += 1
                    continue
                n_pages_used += 1

                for i, text in enumerate(split_page(page["text"])):
                    record = {
                        "chunk_id": f"{ref}-p{page['page']:04d}-c{i}",
                        "ref": ref,
                        "page": page["page"],
                        "n_pages": page["n_pages"],
                        "chunk_index": i,
                        "text": text,
                        "n_chars": len(text),
                        "project_name": meta["project_name"],
                        "country": meta["country"],
                        "theme": meta["theme"],
                        "project_size": meta["project_size"],
                        "ess_category": meta["ess_category"],
                        "funding_usd": float(meta["funding_usd"]),
                        "project_page": meta["project_page"],
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sizes.append(len(text))
                    per_doc += 1
                    n_chunks += 1

            print(f"{ref}  {per_doc:>4} chunks")

    sizes.sort()
    print(f"\n{n_chunks:,} chunks from {n_pages_used:,} pages "
          f"({n_pages_skipped} pages skipped as empty)")
    print(f"chars per chunk: min {sizes[0]} | "
          f"median {sizes[len(sizes) // 2]} | max {sizes[-1]}")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")

if __name__ == "__main__":
    main()