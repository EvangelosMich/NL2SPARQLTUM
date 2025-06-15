"""
natural_language_to_sparql.py
--------------------------------
Turn a plain-English question into a SPARQL query for Wikidata.

How it works
============

1. **parse_label_and_type** – asks Gemini to pull out
   • *label* – the string to feed into Wikidata search  
   • *type*  – a coarse class hint (“film”, “book”, …)

2. **search_wikidata_entity** – looks the label up in Wikidata,
   using the optional *type* hint to make sure we get
   *The Lord of the Rings (film)* (`Q229390`) instead of the book.

3. **natural_language_to_sparql** – asks Gemini to
   convert the original NL question plus the found entity id
   into a SPARQL query.

Replace `YOUR_GEMINI_API_KEY` with your key (don’t hard-code it!).
Run the script and type a question such as

    Who directed the Lord of the Rings movie?

You should see the corresponding SPARQL.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Tuple

import requests
import google.generativeai as genai

# ---------------------------------------------------------------------
# 0.  Gemini setup
# ---------------------------------------------------------------------

genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")


# ---------------------------------------------------------------------
# 1.  Helpers for entity extraction
# ---------------------------------------------------------------------

def parse_label_and_type(nl_query: str) -> Tuple[str, str | None]:
    """
    Ask Gemini to pull out:
      • label – short search string for Wikidata
      • type  – coarse class hint (film, book, person, city, …) or None
    """
    prompt = f"""
You will receive a user question.
Extract:
  "label" – the few words that should be looked up in Wikidata; no quotes.
  "type"  – one of film, book, person, city, unknown (pick the best match).

Return *only* valid JSON like:
{{"label": "...", "type": "..."}}

Question:
{nl_query}
"""
    resp = model.generate_content(prompt).text

    try:
        data = json.loads(resp)
        label = data.get("label", "").strip()
        typ   = data.get("type", "").strip().lower()
        if not label:
            raise ValueError
        if typ == "unknown":
            typ = None
        return label, typ
    except Exception:
        # Fallback: treat the whole question as label
        return nl_query.strip(), None


# ---------------------------------------------------------------------
# 2.  Improved Wikidata search
# ---------------------------------------------------------------------

# Mapping “instance of” (P31) classes we care about
INSTANCE_OF = {
    "film":  "Q11424",
    "movie": "Q11424",
    "book":  "Q571",
    "novel": "Q8261",
    "city":  "Q515",
    "person": "Q5",
}

def search_wikidata_entity(
    label: str,
    wanted_type: str | None = None,
    limit: int = 20,
    session: requests.Session | None = None,
) -> str | None:
    """
    Return the best Wikidata Q-id for *label*,
    preferring items whose P31 matches *wanted_type*.
    """
    sess = session or requests.Session()

    # --- search API ---
    search_params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": label,
        "limit": limit,
    }
    res = sess.get("https://www.wikidata.org/w/api.php",
                   params=search_params, timeout=10)
    res.raise_for_status()
    hits: List[Dict[str, Any]] = res.json().get("search", [])

    if not hits:
        return None                      # nothing found at all

    # --- PASS 1: description heuristics --------------------------
    if wanted_type:
        wt = wanted_type.lower()
        positive = {"film", "movie"} if wt in {"film", "movie"} else {wt}
        negative = {"book", "novel"} if wt in {"film", "movie"} else {"film", "movie"}

        for hit in hits:
            desc = hit.get("description", "").lower()
            if any(p in desc for p in positive) and not any(n in desc for n in negative):
                return hit["id"]

    # --- PASS 2: P31 check ---------------------------------------
    if wanted_type and wanted_type.lower() in INSTANCE_OF:
        candidate_ids = [h["id"] for h in hits[:limit]]
        get_params = {
            "action": "wbgetentities",
            "format": "json",
            "props": "claims",
            "ids": "|".join(candidate_ids),
        }
        ents = sess.get("https://www.wikidata.org/w/api.php",
                        params=get_params, timeout=10).json()["entities"]
        target_qid = INSTANCE_OF[wanted_type.lower()]

        for cid in candidate_ids:
            claims = ents[cid].get("claims", {})
            for p31 in claims.get("P31", []):
                if p31["mainsnak"]["datavalue"]["value"]["id"] == target_qid:
                    return cid   # correct instance found

    # --- fallback: first hit -------------------------------------
    return hits[0]["id"]


# ---------------------------------------------------------------------
# 3.  Prompt Gemini for SPARQL
# ---------------------------------------------------------------------

def natural_language_to_sparql(
    nl_query: str,
    entity_id: str | None = None
) -> str:
    """
    Use Gemini to generate SPARQL for *nl_query*,
    optionally feeding the resolved *entity_id*.
    """
    context = f"The main entity is {entity_id}." if entity_id else ""
    prompt = f"""
Convert the following natural-language question into a SPARQL query
that can be executed against the public Wikidata SPARQL endpoint.

Question:
\"\"\"{nl_query}\"\"\"

{context}

Return *only* the SPARQL.
"""
    return model.generate_content(prompt).text.strip()


# ---------------------------------------------------------------------
# 4.  Little CLI
# ---------------------------------------------------------------------

def main() -> None:
    nl = input("Enter your question: ").strip()
    if not nl:
        print("Nothing to do.", file=sys.stderr)
        return

    label, typ = parse_label_and_type(nl)
    print(f"🔎  label = {label!r},  type hint = {typ}")

    qid = search_wikidata_entity(label, wanted_type=typ)
    print(f"✅  Wikidata entity id: {qid}")

    sparql = natural_language_to_sparql(nl, qid)
    print("\nGenerated SPARQL:\n")
    print(sparql)


if __name__ == "__main__":
    main()
