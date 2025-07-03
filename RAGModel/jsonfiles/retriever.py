from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

###########################################################################
# Configuration                                                           #
###########################################################################

SCRIPT_DIR = Path(__file__).resolve().parent

# List the JSON files that hold entity data. Add/remove as you like.
ENTITY_JSON_FILES: List[str] = [
    "capital.json",
    "companies.json",
    "countries.json",
    "event.json",
    "movies.json",
    "public_figures.json",
]

# Path to the natural‑language property file (predicate labels)
EN_JSON_PATH: Path = SCRIPT_DIR / "en.json"

# Sentence‑Transformers model (using CPU by default)
MODEL_NAME: str = "all-MiniLM-L6-v2"

###########################################################################
# Model & helpers                                                         #
###########################################################################

model = SentenceTransformer(MODEL_NAME, device="cpu")


def _load_entities(path: Path) -> List[Dict[str, Any]]:
    """Normalise various entity JSON formats into a common schema."""
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    entities: List[Dict[str, Any]] = []
    for entry in raw:
        entities.append(
            {
                "id": entry.get("wikidata_id")
                or entry.get("uid")
                or entry.get("pk")
                or entry.get("code"),
                "label": entry.get("label") or entry.get("name") or entry.get("title"),
                "description": entry.get("description")
                or entry.get("summary")
                or "",
                "facts": entry.get("facts")
                or entry.get("aliases")
                or entry.get("aka")
                or [],
            }
        )
    return entities


###########################################################################
# Build entity corpus & FAISS index                                       #
###########################################################################

entity_docs: List[Dict[str, Any]] = []
for fname in ENTITY_JSON_FILES:
    fpath = SCRIPT_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Expected JSON file not found: {fpath}")
    entity_docs.extend(_load_entities(fpath))

entity_texts: List[str] = [
    f"{e['label']} ({e['id']}): {e['description']} {' '.join(e.get('facts', []))}"
    for e in entity_docs
]
entity_ids: List[Any] = [e["id"] for e in entity_docs]

entity_embeddings = model.encode(entity_texts, normalize_embeddings=True)
entity_index = faiss.IndexFlatIP(entity_embeddings.shape[1])
entity_index.add(np.asarray(entity_embeddings, dtype=np.float32))

###########################################################################
# Build property index                                                    #
###########################################################################

with EN_JSON_PATH.open("r", encoding="utf-8") as fp:
    properties: Dict[str, str] = json.load(fp)

prop_texts: List[str] = [v for v in properties.values()]  # No "(P84)"
prop_ids: List[str] = list(properties.keys())

prop_embeddings = model.encode(prop_texts, normalize_embeddings=True)
prop_index = faiss.IndexFlatIP(prop_embeddings.shape[1])
prop_index.add(np.asarray(prop_embeddings, dtype=np.float32))

###########################################################################
# Public API                                                              #
###########################################################################

def retrieve_offline_ids(
    query: str,
    *,
    topk_entity: int = 3,
    topk_prop: int = 5,
    prop_threshold: float = 0.2,
    ent_threshold: float = 0.6
) -> Tuple[List[Any], List[str]]:
    """Return the *ids* of the most similar entities and properties."""
    query_emb = model.encode([query], normalize_embeddings=True)

    # Entity search
    ent_scores, ent_idx = entity_index.search(query_emb, topk_entity)
    entity_hits = [entity_ids[i] for score, i in zip(ent_scores[0],ent_idx[0]) if score>=ent_threshold]
    for score, i in zip(ent_scores[0], ent_idx[0]):
        print(f"{entity_texts[i]}: {score:.3f}")

    query_lower = query.lower()


    for e in entity_docs:
     if re.search(r'\b' + re.escape(e["label"].lower()) + r'\b', query_lower) and e["id"] not in entity_hits:
        entity_hits.append(e["id"])
    # Property search
    prop_scores, prop_idx = prop_index.search(query_emb, topk_prop)
    prop_hits = [prop_ids[i] for score, i in zip(prop_scores[0], prop_idx[0])if score>=prop_threshold]
    for score, i in zip(prop_scores[0], prop_idx[0]):
        print(f"{properties[prop_ids[i]]}: {score:.3f}")

    for i, label in enumerate(prop_texts):
        if label.lower() in query_lower and prop_ids[i] not in prop_hits:
            prop_hits.append(prop_ids[i])

    return entity_hits, prop_hits


###########################################################################
# CLI / quick manual test                                                 #
###########################################################################

if __name__ == "__main__":
    print("Loaded", len(entity_docs), "entities and", len(prop_ids), "properties.")
    while True:
        try:
            q = input("\nQuery (blank to exit): ").strip()
            if not q:
                break
            ents, props = retrieve_offline_ids(q)
            print("Entities:", ents)
            print("Props   :", props)
        except (KeyboardInterrupt, EOFError):
            print()
            break
