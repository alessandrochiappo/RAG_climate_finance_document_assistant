import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_CSV = ROOT / "data" / "gcf_rag_data.csv"

COLUMNS = {
    "ref": "ref",
    "projectname": "project_name",
    "countries": "country",
    "theme": "theme",
    "projectsize": "project_size",
    "esscategory": "ess_category",
    "fafinancing": "funding_usd",
    "url": "pdf_url",
    "projectpage": "project_page",
}

def squash(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_CSV, sep=";")
    df.columns = [squash(c) for c in df.columns]

    missing = [k for k in COLUMNS if k not in df.columns]
    if missing:
        raise KeyError(f"manifest is missing {missing}. Found: {list(df.columns)}")

    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    df["ref"] = df["ref"].astype(str).str.strip().str.upper()
    df["funding_usd"] = pd.to_numeric(df["funding_usd"])

    dupes = df.loc[df["ref"].duplicated(), "ref"].tolist()
    assert not dupes, f"duplicate Ref #: {dupes}"
    assert df["ref"].str.fullmatch(r"FP\d{3}").all(), "malformed Ref #"
    assert df["funding_usd"].notna().all(), "missing funding amount"
    assert df["pdf_url"].str.startswith("http").all(), "bad or missing PDF url"

    return df

if __name__ == "__main__":
    m = load_manifest()
    print(f"{len(m)} projects | {m['country'].nunique()} countries")
    print(f"funding: ${m['funding_usd'].min():,.0f} to ${m['funding_usd'].max():,.0f}")
    print()
    print(m[["ref", "country", "theme", "project_size", "funding_usd"]].head().to_string(index=False))