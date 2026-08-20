
import collections
import json

from manifest import ROOT

CHUNKS = ROOT / "data" / "chunks.jsonl"

per_doc = collections.Counter()
contexts = collections.Counter()

for line in CHUNKS.open(encoding="utf-8"):
    record = json.loads(line)
    text = record["text"]
    i = -1
    while True:
        i = text.find("\ufffd", i + 1)
        if i == -1:
            break
        per_doc[record["ref"]] += 1
        contexts[text[max(0, i - 7):i + 7]] += 1

print(f"total occurrences: {sum(per_doc.values()):,}")
print(f"documents affected: {len(per_doc)} of 34")
print(f"by document: {dict(per_doc.most_common())}\n")

for snippet, count in contexts.most_common(20):
    print(f"{count:>5}  {snippet!r}")