"""Question -> the pages most likely to answer it, with citations."""

import argparse
import json
from collections import OrderedDict
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from manifest import ROOT

CHROMA_DIR = ROOT / "data" / "chroma"
PAGES_DIR = ROOT / "data" / "pages"
COLLECTION = "gcf_proposals"

TOP_K = 150
MAX_PAGES = 6

_page_cache = {}


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(
        name=COLLECTION,
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )


def load_page(ref, page):
    if ref not in _page_cache:
        _page_cache[ref] = {
            r["page"]: r["text"]
            for r in (json.loads(line)
                      for line in (PAGES_DIR / f"{ref}.jsonl").open(encoding="utf-8"))
        }
    return _page_cache[ref].get(page, "")


def build_where(country=None, theme=None, size=None, ref=None, min_funding=None):
    clauses = []
    if country:
        clauses.append({"country": country})
    if theme:
        clauses.append({"theme": theme})
    if size:
        clauses.append({"project_size": size})
    if ref:
        clauses.append({"ref": ref.upper()})
    if min_funding:
        clauses.append({"funding_usd": {"$gte": float(min_funding)}})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def select_pages(candidates, max_pages):
    """Guarantee every matched document a seat, then fill by distance.

    Plain distance ranking returned six pages from one proposal when asked to
    compare three -- the model cannot compare what it never sees.

    An earlier version capped pages per document. That fixed coverage but
    evicted FP272 p.15, the Theory of Change page that actually answered the
    question, because two less relevant pages from the same document were
    marginally closer. Optimising the coverage metric made the answer worse.

    Guaranteeing one page per document gets the same coverage without
    displacing strong pages: pass 1 seats each document once, pass 2 fills
    every remaining slot by pure distance.
    """
    ordered = sorted(candidates, key=lambda p: p["best_distance"])

    selected, seen_refs = [], set()
    for page in ordered:
        if len(selected) >= max_pages:
            break
        if page["ref"] not in seen_refs:
            selected.append(page)
            seen_refs.add(page["ref"])

    chosen = {(p["ref"], p["page"]) for p in selected}
    for page in ordered:
        if len(selected) >= max_pages:
            break
        if (page["ref"], page["page"]) not in chosen:
            selected.append(page)

    return sorted(selected, key=lambda p: p["best_distance"])


def retrieve(question, top_k=TOP_K, max_pages=MAX_PAGES, **filters):
    collection = get_collection()
    result = collection.query(
        query_texts=[question],
        n_results=top_k,
        where=build_where(**filters),
        include=["metadatas", "distances"],
    )

    pages = OrderedDict()
    for chunk_id, meta, distance in zip(result["ids"][0],
                                        result["metadatas"][0],
                                        result["distances"][0]):
        key = (meta["ref"], meta["page"])
        if key not in pages:
            pages[key] = {
                "ref": meta["ref"],
                "page": meta["page"],
                "project_name": meta["project_name"],
                "country": meta["country"],
                "theme": meta["theme"],
                "funding_usd": meta["funding_usd"],
                "project_page": meta["project_page"],
                "best_distance": distance,
                "n_hits": 0,
                "chunk_ids": [],
            }
        pages[key]["n_hits"] += 1
        pages[key]["chunk_ids"].append(chunk_id)

    selected = select_pages(list(pages.values()), max_pages)
    for page in selected:
        page["text"] = load_page(page["ref"], page["page"])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--country")
    parser.add_argument("--theme")
    parser.add_argument("--size")
    parser.add_argument("--ref")
    parser.add_argument("--min-funding", type=float, dest="min_funding")
    parser.add_argument("--top-k", type=int, default=TOP_K, dest="top_k")
    args = parser.parse_args()

    pages = retrieve(
        args.question,
        top_k=args.top_k,
        country=args.country,
        theme=args.theme,
        size=args.size,
        ref=args.ref,
        min_funding=args.min_funding,
    )

    print(f"\nQ: {args.question}")
    print(f"{len(pages)} pages from {len({p['ref'] for p in pages})} document(s)\n")

    for i, page in enumerate(pages, start=1):
        print("=" * 78)
        print(f"[{i}] {page['ref']} p.{page['page']}  |  {page['country']}  |  "
              f"{page['theme']}  |  ${page['funding_usd']:,.0f}")
        print(f"    {page['project_name'][:70]}")
        print(f"    distance {page['best_distance']:.3f} | "
              f"{page['n_hits']} chunk(s) matched")
        print("=" * 78)
        print(page["text"][:600].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()