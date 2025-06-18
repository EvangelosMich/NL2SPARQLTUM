import json
from pathlib import Path
from collections import defaultdict

# Update this path if your data is elsewhere
base_path = Path('saved_data')

files = {
    "labels": base_path / "labels",
    "descriptions": base_path / "descriptions",
    "entity_values": base_path / "entity_values",
    "entity_rels": base_path / "entity_rels",
    "aliases": base_path / "aliases",        # optional
    "qualifiers": base_path / "qualifiers",  # optional
}
entity_db = defaultdict(lambda: {
    "id": None,
    "label": None,
    "description": None,
    "aliases": [],
    "facts": []
})

def load_json_lines_from_folder(folder_path: Path):
    if not folder_path.exists():
        print(f"⚠️  Skipping missing folder: {folder_path}")
        return []
    data = []
    for file_path in sorted(folder_path.glob("*.jsonl")):
        with open(file_path, encoding="utf-8") as f:
            data.extend(json.loads(line) for line in f)
    return data
# Load labels
for item in load_json_lines_from_folder(files["labels"]):
    qid = item["qid"]
    entity_db[qid]["id"] = qid
    entity_db[qid]["label"] = item.get("label") or item.get("value")

# Load descriptions
for item in load_json_lines_from_folder(files["descriptions"]):
    qid = item["qid"]
    entity_db[qid]["id"] = qid
    entity_db[qid]["description"] = item.get("description") or item.get("value")

# Load aliases
for item in load_json_lines_from_folder(files["aliases"]):
    qid = item["qid"]
    entity_db[qid]["id"] = qid
    entity_db[qid]["aliases"].append(item.get("alias") or item.get("value"))

# Load entity values (literal facts)
for item in load_json_lines_from_folder(files["entity_values"]):
    qid = item.get("entity_id") or item.get("subject_id")
    if not qid: continue
    fact = f"{item['property_id']}: {item['value']}"
    entity_db[qid]["facts"].append(fact)

# Load relationships (entity-entity links)
for item in load_json_lines_from_folder(files["entity_rels"]):
    qid = item.get("subject_id") or item.get("entity_id")
    if not qid: continue
    fact = f"{item['property_id']}: {item['object_id']}"
    entity_db[qid]["facts"].append(fact)

# Save as merged JSON
output_path = base_path / "entity_docs.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(list(entity_db.values()), f, ensure_ascii=False, indent=2)

print(f"✅ Merged entity database saved to {output_path}")