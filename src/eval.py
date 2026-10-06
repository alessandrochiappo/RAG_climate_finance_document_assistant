import json
import re
import time

from answer import MODEL, answer
from manifest import ROOT

OUT_PATH = ROOT / "data" / "eval_results.json"

CITATION = re.compile(r"\[(FP\d{3}),\s*((?:p\.)?\s*\d+(?:\s*,\s*(?:p\.)?\s*\d+)*)\]")


def cited_pages(text):
    """Expand [FP170, p.13, 39, 41] into FP170:13, FP170:39, FP170:41.

    The model sometimes puts several pages in one bracket. The previous pattern
    required "p." immediately before a single number, so multi-page citations
    matched nothing at all and any invented page hiding inside one was never
    checked. Measured invented-citation counts were therefore too low, not too
    high -- the metric was silently lenient in exactly the direction that
    flatters the system.
    """
    out = set()
    for ref, pages in CITATION.findall(text):
        for number in re.findall(r"\d+", pages):
            out.add(f"{ref}:{number}")
    return out

REFUSAL_MARKERS = ("no information", "do not contain", "does not contain",
                   "not contain", "no passage", "cannot answer", "not mention",
                   "not provide", "close enough match", "is not available")

ANSWER_KEY = [
    {
        "question": "What is the benefit-cost ratio of this project?",
        "filters": {"ref": "FP171"},
        "expected_pages": ["FP171:85", "FP171:68"],
        "expected_refs": [],
        "expected_text": ["5.7"],
        "should_refuse": False,
        "type": "exact figure",
    },
    {
        "question": "Why were early warning systems chosen over structural flood defences?",
        "filters": {"ref": "FP272"},
        "expected_pages": ["FP272:11", "FP272:15"],
        "expected_refs": [],
        "expected_text": [],
        "should_refuse": False,
        "type": "reasoning / false premise",
    },
    {
        "question": "What carbon price assumption does FP154 use in its economic analysis?",
        "filters": {"ref": "FP154"},
        "expected_pages": ["FP154:70", "FP154:76"],
        "expected_refs": [],
        "expected_text": ["36.30"],
        "should_refuse": False,
        "type": "vocabulary mismatch",
    },
    {
        "question": "What are the four glacial lakes targeted for lowering?",
        "filters": {"ref": "FP272"},
        "expected_pages": [],
        "expected_refs": [],
        "expected_text": ["Thulagi", "Lumding"],
        "should_refuse": False,
        "type": "named entities",
    },
    {
        "question": "How do the Nepal projects differ in their approach to flood risk?",
        "filters": {"country": "Nepal"},
        "expected_pages": [],
        "expected_refs": ["FP272", "FP118", "FP131"],
        "expected_text": [],
        "should_refuse": False,
        "type": "cross-document",
    },
    {
        "question": "What adaptation measures do the Tajikistan projects use?",
        "filters": {"country": "Tajikistan"},
        "expected_pages": [],
        "expected_refs": ["FP040", "FP233", "FP067", "FP075"],
        "expected_text": [],
        "should_refuse": False,
        "type": "filtered multi-doc",
    },
    {
        "question": "How do they stop water running out during the dry season?",
        "filters": {},
        "expected_pages": [],
        "expected_refs": [],
        "expected_text": [],
        "should_refuse": False,
        "type": "obliquely worded (read-only)",
    },
    {
        "question": "What is the capital of Brazil?",
        "filters": {},
        "expected_pages": [],
        "expected_refs": [],
        "expected_text": [],
        "should_refuse": True,
        "type": "out of corpus",
    },
]


def score_retrieval(pages, expected_pages, expected_refs):
    got = [f"{p['ref']}:{p['page']}" for p in pages]
    hits = [i for i, ref in enumerate(got, start=1) if ref in expected_pages]
    refs_seen = {p["ref"] for p in pages}
    return {
        "retrieved": got,
        "refs_seen": sorted(refs_seen),
        "recall": 1.0 if hits else 0.0,
        "rr": 1.0 / hits[0] if hits else 0.0,
        "rank": hits[0] if hits else None,
        "coverage": (len(refs_seen & set(expected_refs)) / len(expected_refs)
                     if expected_refs else None),
        "missing_refs": sorted(set(expected_refs) - refs_seen),
    }


def score_citations(text, pages, expected_pages):
    cited = cited_pages(text)
    supplied = {f"{p['ref']}:{p['page']}" for p in pages}
    return {
        "cited": sorted(cited),
        "invented": sorted(cited - supplied),
        "off_target": sorted(cited - set(expected_pages)) if expected_pages else [],
    }


def looks_like_refusal(text):
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def main():
    results = []
    print(f"evaluating with model: {MODEL} (pinned, no fallbacks)\n")

    for i, case in enumerate(ANSWER_KEY, start=1):
        print(f"[{i}/{len(ANSWER_KEY)}] {case['question'][:62]}")

        try:
            text, pages = answer(case["question"], models=[MODEL], **case["filters"])
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__} -- skipping")
            text, pages = f"[ERROR: {type(exc).__name__}]", []

        time.sleep(2)

        results.append({
            **case,
            "model": MODEL,
            "answer": text,
            **score_retrieval(pages, case["expected_pages"], case["expected_refs"]),
            **score_citations(text, pages, case["expected_pages"]),
            "refused": looks_like_refusal(text),
            "refusal_correct": looks_like_refusal(text) == case["should_refuse"],
            "missing_expected_text": [t for t in case["expected_text"] if t not in text],
        })

    labelled = [r for r in results if r["expected_pages"]]
    multi_doc = [r for r in results if r["expected_refs"]]
    answerable = [r for r in results if not r["should_refuse"]]
    refusals = [r for r in results if r["should_refuse"]]

    print(f"\n{'=' * 78}")
    print(f"MODEL: {MODEL}")

    print("\nRETRIEVAL  (questions with a labelled ground-truth page)")
    if labelled:
        recall = sum(r["recall"] for r in labelled) / len(labelled)
        mrr = sum(r["rr"] for r in labelled) / len(labelled)
        print(f"  Recall@6   {recall:.2f}   "
              f"({sum(r['recall'] for r in labelled):.0f}/{len(labelled)})")
        print(f"  MRR        {mrr:.2f}")
        for r in labelled:
            mark = "OK  " if r["recall"] else "MISS"
            print(f"    {mark} rank {str(r['rank']):>4}  {r['question'][:52]}")

    print("\nDOCUMENT COVERAGE  (cross-document questions)")
    if multi_doc:
        cov = sum(r["coverage"] for r in multi_doc) / len(multi_doc)
        print(f"  mean coverage  {cov:.2f}")
        for r in multi_doc:
            n_hit = len(set(r["expected_refs"]) & set(r["refs_seen"]))
            print(f"    {n_hit}/{len(r['expected_refs'])}  {r['question'][:46]}")
            print(f"          got {r['refs_seen']}  missing {r['missing_refs']}")

    print("\nCITATIONS")
    invented = [r for r in answerable if r["invented"]]
    off_target = [r for r in labelled if r["off_target"]]
    print(f"  invented (page never supplied)  {len(invented)}")
    for r in invented:
        print(f"    {r['question'][:50]}  ->  {r['invented']}")
    print(f"  off-target (supplied, not the answer page)  {len(off_target)}")
    for r in off_target:
        print(f"    {r['question'][:50]}  ->  {r['off_target']}")

    print("\nEXPECTED TEXT")
    for r in results:
        if r["expected_text"]:
            missing = r["missing_expected_text"]
            status = "OK" if not missing else f"MISSING {missing}"
            print(f"  {status:<28} {r['question'][:46]}")

    print("\nREFUSALS")
    print(f"  correctly refused  {sum(r['refusal_correct'] for r in refusals)}"
          f"/{len(refusals)}")
    leaked = [r for r in answerable if r["refused"]]
    if leaked:
        print(f"  refused when it should have answered  {len(leaked)}")
        for r in leaked:
            print(f"    {r['question'][:60]}")

    print(f"\n{'=' * 78}")
    print("ANSWERS -- read these. The metrics do not judge factual correctness.\n")
    for i, r in enumerate(results, start=1):
        print(f"[{i}] ({r['type']})  rank {r['rank']}  coverage {r['coverage']}")
        print(f"    Q: {r['question']}")
        print(f"    A: {r['answer'][:500]}")
        print()

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"full results -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()