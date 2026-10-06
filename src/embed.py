import argparse
import json
import time

import chromadb
import numpy as np
from chromadb.utils import embedding_functions

from manifest import ROOT

CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "gcf_proposals"

BATCH = 256

META_FIELDS = ("ref", "page", "n_pages", "chunk_index", "project_name",
               "country", "theme", "project_size", "ess_category",
               "funding_usd", "project_page")


def normalise(vectors) -> list[list[float]]:
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", help="embed only this document, e.g. FP075")
    args = parser.parse_args()

    print("loading local embedding model...")
    started = time.time()
    embed = embedding_functions.DefaultEmbeddingFunction()
    print(f"model ready in {time.time() - started:.1f}s\n")

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embed,
    )

    records = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
    if args.ref:
        records = [r for r in records if r["ref"] == args.ref]
    if not records:
        raise SystemExit(f"no chunks found for {args.ref!r}")

    existing = set(collection.get(include=[])["ids"])
    todo = [r for r in records if r["chunk_id"] not in existing]

    print(f"{len(records):,} chunks selected | {len(todo):,} to embed "
          f"| {len(records) - len(todo):,} already done\n")
    if not todo:
        print("nothing to do -- collection is already complete")
        return

    started = time.time()
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]

        vectors = normalise(embed([r["text"] for r in batch]))

        collection.add(
            ids=[r["chunk_id"] for r in batch],
            embeddings=vectors,
            metadatas=[{k: r[k] for k in META_FIELDS} for r in batch],
        )

        done = min(i + BATCH, len(todo))
        elapsed = time.time() - started
        rate = done / elapsed if elapsed else 0
        eta = (len(todo) - done) / rate / 60 if rate else 0
        print(f"  {done:>6,}/{len(todo):,}  "
              f"({rate:.0f} chunks/s, ~{eta:.1f} min left)")

    print(f"\ncollection now holds {collection.count():,} vectors")
    print(f"index directory: {CHROMA_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
