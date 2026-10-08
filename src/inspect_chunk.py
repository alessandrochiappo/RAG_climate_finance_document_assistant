"""How bad is the oversize problem, and what do the offenders look like?"""

import json

from manifest import ROOT

CHUNKS = ROOT / "data" / "chunks.jsonl"
LIMIT = 1500  

records = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
big = sorted((r for r in records if r["n_chars"] > LIMIT),
             key=lambda r: -r["n_chars"])

print(f"{len(big)} of {len(records):,} chunks are over {LIMIT} chars "
      f"({100 * len(big) / len(records):.1f}%)\n")

for r in big[:5]:
    print(f"--- {r['chunk_id']}  {r['n_chars']} chars ---")
    print(r["text"][:400].replace("\n", " "))
    print()