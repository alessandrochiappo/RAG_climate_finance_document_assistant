import os
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from answer import MAX_DISTANCE, answer          
from manifest import load_manifest               
from retrieve import retrieve                    

st.set_page_config(page_title="GCF Proposal Assistant", page_icon="📄",
                   layout="wide")

try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

EXAMPLES = [
    ("Protect rice fields",
     "How are rice fields protected from climate impacts?"),
    ("Dry-season water",
     "How do they stop water running out during the dry season?"),
    ("Flood defences",
     "Why were early warning systems chosen over structural flood defences?"),
    ("Capital of Brazil",
     "What is the capital of Brazil?"),
]


@st.cache_data
def manifest():
    return load_manifest()


m = manifest()
PDF_URL = dict(zip(m["ref"], m["pdf_url"]))

CITATION_RE = re.compile(r"\[(FP\d{3}),\s*((?:p\.)?\s*\d+(?:\s*,\s*(?:p\.)?\s*\d+)*)\]")


def linkify(text):
    """Turn [FP171, p.85] in the model's answer into a link to that PDF page.

    PDF viewers honour a #page=N fragment, so a citation becomes a jump straight
    to the page it cites -- the reader checks the claim in one click instead of
    taking it on trust. The model still only ever emits the label; the URL is
    looked up from the manifest by ref.
    """
    def replace(match):
        ref, pages = match.group(1), match.group(2)
        url = PDF_URL.get(ref)
        if not url:
            return match.group(0)
        numbers = re.findall(r"\d+", pages)
        links = ", ".join(f"[p.{n}]({url}#page={n})" for n in numbers)
        return f"\\[{ref}, {links}\\]"

    return CITATION_RE.sub(replace, text)


st.title("GCF Funding Proposal Assistant")
st.caption(
    f"{len(m)} Green Climate Fund proposals · {m['country'].nunique()} countries · "
    f"${m['funding_usd'].sum() / 1e9:.2f}B in approved finance · "
    "every answer cited to document and page"
)

tab_ask, tab_about, tab_portfolio, tab_compare = st.tabs(
    ["Ask", "About", "Portfolio", "Compare projects"])

with tab_ask:
    with st.sidebar:
        st.header("Filters")
        st.caption("Applied before the search, not after — they narrow what is "
                   "searched rather than what is shown.")

        country = st.selectbox("Country", ["Any"] + sorted(m["country"].unique()))
        theme = st.selectbox("Theme", ["Any"] + sorted(m["theme"].unique()))
        size = st.selectbox("Project size", ["Any"] + sorted(m["project_size"].unique()))
        min_funding = st.slider("Minimum funding (US$m)", 0, 175, 0, step=5)

        filters = {
            "country": None if country == "Any" else country,
            "theme": None if theme == "Any" else theme,
            "size": None if size == "Any" else size,
            "min_funding": None if min_funding == 0 else min_funding * 1e6,
        }

        eligible = m.copy()
        if filters["country"]:
            eligible = eligible[eligible["country"] == filters["country"]]
        if filters["theme"]:
            eligible = eligible[eligible["theme"] == filters["theme"]]
        if filters["size"]:
            eligible = eligible[eligible["project_size"] == filters["size"]]
        if filters["min_funding"]:
            eligible = eligible[eligible["funding_usd"] >= filters["min_funding"]]
        st.metric("Proposals in scope", len(eligible))

    st.subheader("Ask a question")

    if "question" not in st.session_state:
        st.session_state.question = ""

    cols = st.columns(len(EXAMPLES))
    for col, (label, example) in zip(cols, EXAMPLES):
        if col.button(label, use_container_width=True):
            st.session_state.question = example

    question = st.text_input("Question", key="question",
                             placeholder="e.g. how is climate risk assessed?",
                             label_visibility="collapsed")

    if st.button("Ask", type="primary") and question.strip():
        if eligible.empty:
            st.warning("No proposals match those filters.")
        else:
            with st.spinner("Searching 21,525 passages…"):
                try:
                    text, pages = answer(question, **filters)
                except Exception as exc:
                    st.error(f"The language model is unavailable right now "
                             f"({type(exc).__name__}). Retrieval still works — "
                             f"the sources below are what it found.")
                    text, pages = None, retrieve(question, **filters)

            if text:
                st.markdown(linkify(text))

            if pages:
                if pages[0]["best_distance"] > MAX_DISTANCE:
                    st.info(
                        f"Closest passage scored {pages[0]['best_distance']:.2f}; "
                        f"anything above {MAX_DISTANCE} is treated as no match. "
                        "The refusal is decided in code, before the model is called."
                    )

                st.divider()
                st.caption(f"{len(pages)} pages from "
                           f"{len({p['ref'] for p in pages})} proposal(s) — "
                           "click any citation to open the PDF at that page")

                for page in pages:
                    label = (f"[{page['ref']}, p.{page['page']}] · "
                             f"{page['country']} · {page['theme']} · "
                             f"distance {page['best_distance']:.3f}")
                    with st.expander(label):
                        st.caption(page["project_name"])
                        url = PDF_URL.get(page["ref"], page["project_page"])
                        st.markdown(
                            f"[Open {page['ref']} PDF at page {page['page']}]"
                            f"({url}#page={page['page']}) · "
                            f"[Project page]({page['project_page']})"
                        )
                        st.write(page["text"][:2500])

with tab_compare:
    st.subheader("Compare proposals side by side")

    labels = {f"{r.ref} — {r.project_name[:60]}": r.ref for r in m.itertuples()}
    picked = st.multiselect("Select 2–4 proposals", list(labels), max_selections=4)
    refs = [labels[p] for p in picked]

    if refs:
        table = (m[m["ref"].isin(refs)]
                 .set_index("ref")[["project_name", "country", "theme",
                                    "project_size", "ess_category", "funding_usd"]]
                 .T)
        table.index = ["Project", "Country", "Theme", "Size", "ESS category",
                       "Funding (US$)"]
        st.dataframe(table, use_container_width=True)
        st.caption("These fields come from the manifest, not the language model — "
                   "exact by construction, no retrieval involved.")

        st.divider()
        st.markdown("**Ask the same question of each**")
        shared = st.text_input(
            "Question", key="compare_q",
            placeholder="e.g. how does this project assess climate risk?",
            label_visibility="collapsed")

        if st.button("Compare answers", type="primary") and shared.strip():
            for ref, col in zip(refs, st.columns(len(refs))):
                with col:
                    st.markdown(f"**{ref}**")
                    try:
                        with st.spinner(""):
                            text, pages = answer(shared, ref=ref)
                        st.markdown(linkify(text))
                        url = PDF_URL.get(ref, "")
                        st.markdown(" · ".join(
                            f"[p.{p['page']}]({url}#page={p['page']})"
                            for p in pages[:4]))
                    except Exception as exc:
                        st.error(type(exc).__name__)

with tab_portfolio:
    st.subheader("The portfolio")
    st.caption("Sorted, filtered and totalled by pandas over the manifest — no "
               "retrieval, no language model. Ranking and arithmetic are exact "
               "by construction; a vector index cannot compare numbers, so it "
               "is never asked to.")

    view = (m.sort_values("funding_usd", ascending=False)
             [["ref", "project_name", "country", "theme", "project_size",
               "ess_category", "funding_usd"]]
             .rename(columns={"ref": "Ref", "project_name": "Project",
                              "country": "Country", "theme": "Theme",
                              "project_size": "Size",
                              "ess_category": "ESS category",
                              "funding_usd": "Funding (US$)"}))

    largest = view.iloc[0]
    smallest = view.iloc[-1]

    a, b, c = st.columns(3)
    a.metric("Largest", largest["Ref"], f"${largest['Funding (US$)'] / 1e6:,.0f}M")
    b.metric("Smallest", smallest["Ref"], f"${smallest['Funding (US$)'] / 1e6:,.1f}M")
    c.metric("Total approved", f"${m['funding_usd'].sum() / 1e9:.2f}B",
             f"{len(m)} proposals")

    st.dataframe(view, use_container_width=True, hide_index=True,
                 column_config={"Funding (US$)": st.column_config.NumberColumn(
                     format="$%d")})

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("**By country**")
        st.dataframe(
            m.groupby("country")
             .agg(proposals=("ref", "count"), funding=("funding_usd", "sum"))
             .sort_values("funding", ascending=False),
            use_container_width=True)
    with right:
        st.markdown("**By theme and size**")
        st.dataframe(
            m.pivot_table(index="theme", columns="project_size", values="ref",
                          aggfunc="count", fill_value=0),
            use_container_width=True)


with tab_about:
    st.markdown(f"""
### What this is

Every project the Green Climate Fund approves comes with a funding proposal
explaining what the money is for, what could go wrong, and why the plan should
work. They run **100 to 300 pages each**. This tool holds {len(m)} of them,
${m['funding_usd'].sum() / 1e9:.2f} billion of approved climate finance across
{m['country'].nunique()} countries, and lets you ask questions in ordinary
English instead of reading them.

Ask *"how do they stop water running out during the dry season?"* and you get an
answer in a few seconds, with a link to the exact page of the exact PDF each part
of the answer came from.

### Why it isn't just a search box

A keyword search looks for your words. These documents don't use your words
they say *"dry-season water deficit"* and *"supplementary irrigation"*, which is
not what anyone types.

So the search works on **meaning** rather than spelling. Every passage in all
{len(m)} documents has been converted into a string of numbers describing what it
is about, and your question is converted the same way. The passages whose numbers
sit closest to your question's are the ones that come back, whether or not they
share a single word with what you typed.

### How to know the answer isn't invented

This is the part that matters, and it's worth being precise about.

**The page number is stamped on by code, not written by the AI.** When the PDFs
were first read, each page was saved with its number attached — `FP171, page 85`.
The AI is handed passages that are already labelled and told to copy the label
across. It never decides what page something came from, and it never sees a page
it wasn't given.

**Every citation is a link.** Click `[FP171, p.85]` in any answer and the real
GCF document opens at page 85, so you can read the sentence yourself.

**And you should click.** Testing found that the AI occasionally writes a page
number it was never handed, rare, but it happens. The system is built to make
checking take one second, not to make checking unnecessary. Anything claiming the
second thing is overselling.

### Two different tools behind one page

Some questions are about **numbers**: which project is the biggest, how much went
to Nepal, how many proposals are there. Those are answered by a spreadsheet-style
lookup over a table of project facts, on the **Portfolio** tab. The answer is
exact, instantly, every time, because it is arithmetic rather than reading.

Other questions are about **meaning**: why was this approach chosen, how is risk
assessed. Those go to the search described above.

Keeping them apart is deliberate. Asking a meaning-based search to rank numbers
would be slower and would sometimes be wrong, so it is never asked to.

### What it does badly

Tested, on purpose:

- **It won't compare documents for you.** Ask *"how do the Nepal projects
  differ?"* and it finds all three Nepal proposals, then declines to draw the
  comparison. Everything it needs is in front of it; it just won't do that step.
- **It occasionally invents a page number.** As above — rare, real, not hidden.
- **It can't see pictures.** Diagrams are read as their caption only, and a few of
  those diagrams carry the most important logic in the document.
- **Tables lose their shape.** A budget table becomes a run of numbers that no
  longer line up with their headings.
- **Typing "which project is biggest" into the question box won't work.** That's a
  Portfolio-tab question. Sending it to the right place automatically is the next
  thing to build.

### The technical version

Pipeline: PDFs → page-indexed text → 21,525 chunks of roughly 800 characters →
384-dimension embeddings (`all-MiniLM-L6-v2`, running locally) → Chroma vector
store. Retrieval matches chunks by cosine distance, then sends the **full pages**
those chunks came from to Gemini, which writes the answer over them. Chunks never
cross a page boundary, because a citation that spans two pages would be a lie.

Out-of-scope questions are refused in code, before any API call, when the nearest
passage exceeds a distance threshold set from measured data — not by asking the
model whether it knows.

The evaluation was run four times with no code changes between runs:

| Metric | Result | Stable across runs? |
|---|---|---|
| Recall@6 | 1.00 | identical, 4/4 |
| MRR | 0.83 | identical, 4/4 |
| Document coverage | 1.00 | identical, 4/4 |
| Exact figures reproduced | correct | 4/4 |
| Correct refusal, out of corpus | 1/1 | 4/4 |
| Invented citations | rare, non-zero | varies |
| Cross-document synthesis | fails | 0/4 |

Recall@6 of 1.00 means the page holding the answer was among the six sent to the
model on every labelled question. Everything upstream of the model is
deterministic — search runs locally, so those three metrics are bit-identical
every run. Only the generation step varies, and only in citation precision.

Retrieval metrics follow BEIR; generation follows RAGAS faithfulness. Ground
truth was established by keyword search, deliberately independent of the vector
index being measured — and corrected three times when the answer key, not the
system, turned out to be wrong.

Because search runs locally, it costs nothing and cannot hit a rate limit. A
hosted model is called once per question, only to write the answer.
""")
