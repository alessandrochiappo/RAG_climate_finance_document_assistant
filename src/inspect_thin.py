import json

from manifest import ROOT

PAGES_DIR = ROOT / "data" / "pages"
MIN_PAGE_CHARS = 50

for path in sorted(PAGES_DIR.glob("*.jsonl")):
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r["n_chars"] < MIN_PAGE_CHARS:
            print(f"{r['ref']}  p.{r['page']:>3}/{r['n_pages']:>3}  "
                  f"{r['n_chars']:>3} chars   {r['text']!r}")