import requests
import openai
import json
import re

# === CONFIGURATION ===
openai.api_key = (
    "sk-proj-kCLczx7_1zyWErWOqCnavSTcYpjt5bhizlzjey4qTcJlk3Mljr-"
    "zBwmpbR0kwqbMZXr9ccS33ST3BlbkFJ7gril2RheKhSuhh3qBET0GKYb4kKwBEbqM85dl2"
    "FtrmilTw0iChuI5B96BVxX25tvg3ZkIRREA"
)
SERPER_API_KEY = "b4b565c31a58bca9cb7e6201a23a007b3800a537"

# === LLM CONVERSATION CONTEXT ===
message_history = []

def call_openai(prompt, role="user", reset=False):
    global message_history
    if reset:
        message_history = []
        message_history.append({
            "role": "system",
            "content": "You are a helpful assistant that converts natural language into SPARQL queries for Wikidata. You extract concepts, map them to roles, and generate SPARQL queries step-by-step."
        })
    message_history.append({"role": role, "content": prompt})
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=message_history,
        temperature=0
    )
    reply = response.choices[0].message.content.strip()
    message_history.append({"role": "assistant", "content": reply})
    return reply


# === STEP 1: Extract semantic roles ===

def convert_query_to_wikidata_search(output):
    prompt = f"""
Your job is to extract only the **semantic concepts** from the user's natural language query that should be searched on Wikidata to build a SPARQL query.

Ignore helper words like "give", "me", "show", "list", etc. Focus on real-world **concepts**: things, people, places, classes, etc.

Return as a Python list of dictionaries:
[
  {{ "term": "cat", "role": "class" }},
  {{ "term": "image", "role": "media" }}
]

Query: "{output}"
Only return the list.
"""
    text = call_openai(prompt)

    if text.startswith("```"):
        text = text.strip("```").strip()
        if text.startswith("python"):
            text = text[len("python"):].strip()

    terms = eval(text)

    if isinstance(terms, list) and all("term" in t and "role" in t for t in terms):
        return terms
    else:
        raise ValueError("LLM output did not follow expected structure")


# === STEP 1.5: Normalize roles ===
def normalize_roles(terms):
    for entry in terms:
        role = entry["role"].lower()
        term = entry["term"].lower()

        if role == "property" and term in {"male", "female", "color"}:
            entry["role"] = "value"

        if role == "class" and "winner" in term:
            entry["role"] = "human"

        if entry["role"].endswith("s"):
            entry["role"] = entry["role"][:-1]

    return terms


# === STEP 2: Use Serper to search Wikidata ===
def query_search_api(terms):
    url = "https://google.serper.dev/search"
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    term_results = {}
    for term in terms:
        payload = json.dumps({"q": term + " site:wikidata.org"})
        response = requests.post(url, headers=headers, data=payload)
        data = response.json()
        term_results[term] = data.get("organic", [])
    
    return term_results


# === STEP 3: Extract Q-IDs ===
def extract_ids_per_term(results_dict, max_ids_per_term=4):
    all_ids = []
    seen = set()

    for term, results in results_dict.items():
        term_lower = term.lower()
        count = 0
        for result in results:
            link = result.get("link", "")
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            match = re.search(r"wikidata\.org/wiki/(Q\d+)", link)
            if match:
                qid = match.group(1)
                if qid not in seen:
                    relevance = term_lower in title or term_lower in snippet
                    if relevance or count == 0:
                        seen.add(qid)
                        all_ids.append(qid)
                        count += 1
                if count == max_ids_per_term:
                    break
    return all_ids


# === STEP 4: Retrieve labels and descriptions ===
def get_wikidata_descriptions(qids, language='en'):
    if not qids:
        return []

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "format": "json",
        "props": "descriptions|labels",
        "languages": language
    }
    response = requests.get(url, params=params)
    data = response.json()
    result = []

    for qid in qids:
        entity = data["entities"].get(qid, {})
        label = entity.get("labels", {}).get(language, {}).get("value", "")
        description = entity.get("descriptions", {}).get(language, {}).get("value", "")
        result.append({"id": qid, "label": label, "description": description})
    
    return result


# === STEP 5: Generate SPARQL ===
def natural_language_to_sparql(nl_query, entities, role_map):
    context_lines = [f"{e['label']} ({e['id']}): {e['description']}" for e in entities]
    context = "Here are some candidate entities from Wikidata:\n" + "\n".join(context_lines)

    role_instructions = {
        "color": "Use property P462 (color)",
        "class": "Use property P31 (instance of), or P279 (subclass of)",
        "occupation": "Use property P106 (occupation)",
        "country": "Use P495 (country of origin) or P27 (citizenship)",
        "object": "Use contextually appropriate properties like P57 (director), P50 (author)",
        "media": "Use P18 (image), P1651 (YouTube ID), etc.",
        "human": "Use P31 wd:Q5 for humans",
        "gender": "Use P21 (sex or gender)",
        "value": "Use P21 for gender, P462 for color, etc."
    }

    mapped_roles = "\n".join(
        f"{r['term']} → {r['role']}: {role_instructions.get(r['role'], 'Use best-fit property')}"
        for r in role_map
    )

    prompt = f"""
You're a SPARQL expert. Given a natural language query and a list of possible Wikidata entities with labels and descriptions, write a SPARQL query to answer it.

Each entity is identified by its Q-ID, label, and description. Carefully use these clues to pick correct subjects and properties.

{context}

Here are the extracted terms and their roles:
{mapped_roles}

Use meaningful variables like ?item, ?person, etc.
Use specific properties: P21 (gender), P166 (award), P106 (occupation), P27 (citizenship), etc.

Now generate a SPARQL query for:
"{nl_query}"

Return only the SPARQL query.
"""
    return call_openai(prompt)


# === INTERACTIVE LOOP WITH MEMORY ===
def interactive_loop():
    print("🔁 Natural Language → SPARQL Assistant (Type 'reset' to restart or 'exit' to quit)")

    last_query = None
    last_roles = None
    last_entities = None

    while True:
        user_input = input("\n📝 Your query: ").strip()
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reset":
            call_openai("", reset=True)
            last_query = last_roles = last_entities = None
            print("✅ Conversation reset.")
            continue

        use_last_context = False
        try:
            # Try to extract roles
            roles = convert_query_to_wikidata_search(user_input)
            roles = normalize_roles(roles)

            # If no roles were found, fallback to last context
            if not roles:
                raise ValueError("No roles extracted")

            # Fresh query, update state
            last_query = user_input
            last_roles = roles
            terms = [r["term"] for r in roles]
            search_results = query_search_api(terms)
            qids = extract_ids_per_term(search_results)
            entities = get_wikidata_descriptions(qids)
            last_entities = entities

            print(f"\n🔍 Extracted terms with roles: {roles}")
            print("\n🔎 Top Wikidata entities:")
            for e in entities:
                print(f"- {e['label']} ({e['id']}): {e['description']}")

        except Exception:
            # Use previous state
            if not (last_query and last_roles and last_entities):
                print("❌ Not enough previous context to interpret this follow-up.")
                continue
            print("↪️ Using previous context to interpret follow-up.")
            user_input = f"{last_query}. User adds: {user_input}"
            roles = last_roles
            entities = last_entities

        # Generate SPARQL
        try:
            sparql_query = natural_language_to_sparql(user_input, entities, roles)
            print("\n🧠 Generated SPARQL Query:\n")
            print(sparql_query)
        except Exception as e:
            print(f"❌ LLM Error: {e}")



# === MAIN ===
if __name__ == "__main__":
    interactive_loop()
