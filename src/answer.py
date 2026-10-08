import argparse
import os
import textwrap
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from manifest import ROOT
from retrieve import retrieve

MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash"]

MAX_DISTANCE = 0.55
MAX_ATTEMPTS = 4
TRANSIENT = ("503", "500", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")

SYSTEM_PROMPT = """You answer questions about Green Climate Fund funding proposals.

Rules:
1. Use ONLY the passages provided. Do not use outside knowledge about these
   projects, climate finance, or the countries involved.
2. After every factual claim, cite the source in square brackets exactly as it
   is labelled, e.g. [FP171, p.85]. Never invent a page number.
3. Lead with what the passages DO say. Never open by describing what is absent.
   If they contain nothing relevant at all, say so plainly and stop. If they
   contain a partial answer, give it, and put any caveat in one short sentence
   at the end.
4. The question's wording does not have to match the document's. Match on
   meaning, not words. A section headed "Strategic objectives" answers a
   question about objectives; "social value of carbon" answers a question
   about carbon price; "dry-season water deficit" and "supplementary
   irrigation" answer a question about water running out in summer. Everyday
   phrasing of a technical topic is still that topic. A vocabulary mismatch is
   never a reason to say the passages do not cover something.
5. Reproduce figures exactly as they appear in the passages, character for
   character, including decimal points. Never reformat or round a number.
6. If the question assumes a fact the passages contradict, say so, then state
   what the passages do say instead and cite it. Rejecting a premise is never a
   complete answer on its own. A question about something the passages simply do
   not mention is not a false premise.
7. Be concise. Two or three short paragraphs at most.
8. You may compare, contrast and synthesise across the passages you were given.
   A comparison no single passage states is still a valid answer, provided every
   underlying fact carries its own citation. Rule 1 forbids outside knowledge,
   not reasoning over the passages supplied."""


def build_prompt(question, pages):
    blocks = [
        f"--- [{p['ref']}, p.{p['page']}] {p['project_name']} ({p['country']}) ---\n"
        f"{p['text']}"
        for p in pages
    ]
    return "PASSAGES:\n\n" + "\n\n".join(blocks) + f"\n\n---\n\nQUESTION: {question}"


def generate(client, prompt, models=None):
    """Try each model in turn, retrying transient failures before moving on.

    Google's flash models returned 503 UNAVAILABLE for long stretches while this
    was being built. A deployed demo that dies when one model is busy is worse
    than a slightly weaker answer from the next one, so the app walks a chain.

    The eval passes models=[MODEL] instead, because results produced by
    different models are not comparable and the eval needs one pinned model.

    A non-transient error (404 retired model, 400 bad request) re-raises at once
    rather than being retried four times and then blamed on the next model.
    """
    chain = models if models is not None else [MODEL] + FALLBACK_MODELS
    last_error = None

    for model in chain:
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                    ),
                )
                if model != chain[0]:
                    print(f"    (answered by fallback model {model})")
                return response.text, model

            except Exception as exc:
                last_error = exc
                message = str(exc)
                if not any(code in message for code in TRANSIENT):
                    raise
                if attempt == MAX_ATTEMPTS - 1:
                    print(f"    {model} unavailable -- trying next model")
                    break
                wait = min(30, 5 * (2 ** attempt))
                print(f"    {model}: retrying in {wait}s "
                      f"({attempt + 1}/{MAX_ATTEMPTS})")
                time.sleep(wait)

    raise last_error


def answer(question, models=None, **filters):
    pages = retrieve(question, **filters)

    if not pages or pages[0]["best_distance"] > MAX_DISTANCE:
        return ("No passage in the corpus is a close enough match to answer that. "
                "Try rephrasing, or check whether these 34 proposals cover the topic."), pages

    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    text, _model_used = generate(client, build_prompt(question, pages), models=models)
    return text, pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--country")
    parser.add_argument("--theme")
    parser.add_argument("--size")
    parser.add_argument("--ref")
    parser.add_argument("--min-funding", type=float, dest="min_funding")
    parser.add_argument("--pin-model", action="store_true", dest="pin_model",
                        help="use only the primary model, no fallbacks")
    args = parser.parse_args()

    text, pages = answer(
        args.question,
        models=[MODEL] if args.pin_model else None,
        country=args.country,
        theme=args.theme,
        size=args.size,
        ref=args.ref,
        min_funding=args.min_funding,
    )

    print(f"\nQ: {args.question}\n")
    print(textwrap.fill(text, width=90, replace_whitespace=False))
    print(f"\n{'-' * 60}\nPages supplied to the model:")
    for page in pages:
        print(f"  [{page['ref']}, p.{page['page']}]  {page['country']}  "
              f"(distance {page['best_distance']:.3f})")


if __name__ == "__main__":
    main()