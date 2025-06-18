from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2",device="cpu")
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

ENTITY_DOC_PATH = os.path.join(script_dir, "entity_docs.json")
EN_JSON_PATH = os.path.join(script_dir, "en.json")
ENTITY_INDEX_PATH = os.path.join(script_dir, "entity.index")
ENTITY_IDS_PATH = os.path.join(script_dir, "entity_ids.npy")
PROP_INDEX_PATH = os.path.join(script_dir, "prop.index")
PROP_IDS_PATH = os.path.join(script_dir, "prop_ids.npy")

if os.path.exists(ENTITY_INDEX_PATH):
    entity_index = faiss.read_index(ENTITY_INDEX_PATH)
    entity_ids = np.load(ENTITY_IDS_PATH, allow_pickle=True).tolist()
else:
    with open(ENTITY_DOC_PATH) as f:
        entity_docs = json.load(f)

    texts = [
    f"{entry['label']} ({entry['id']}): {entry['description']} {' '.join(entry.get('facts', []))}"
    for entry in entity_docs
    ]
    entity_ids = [entry["id"] for entry in entity_docs]
    entity_embeddings = model.encode(texts, normalize_embeddings=True)

    entity_index = faiss.IndexFlatIP(entity_embeddings.shape[1])
    entity_index.add(np.array(entity_embeddings))

    # Save for next time
    faiss.write_index(entity_index, ENTITY_INDEX_PATH)
    np.save(ENTITY_IDS_PATH, entity_ids)




with open(EN_JSON_PATH,"r") as f:
    properties = json.load(f)

prop_texts = [f"{v} ({k})" for k,v in properties.items()] 
prop_ids = list(properties.keys())

prop_embeddings = model.encode(prop_texts,normalize_embeddings = True)

prop_index = faiss.IndexFlatIP(prop_embeddings.shape[1])
prop_index.add(np.array(prop_embeddings))

def retrieve_offline_ids(query, topk_entity=5, topk_prop=3):
    query_emb = model.encode([query], normalize_embeddings=True)

    # Entity retrieval
    _, entity_idxs = entity_index.search(query_emb, topk_entity)
    found_entities = [entity_ids[i] for i in entity_idxs[0]]

    # Property retrieval
    _, prop_idxs = prop_index.search(query_emb, topk_prop)
    found_props = [prop_ids[i] for i in prop_idxs[0]]

    return found_entities, found_props

