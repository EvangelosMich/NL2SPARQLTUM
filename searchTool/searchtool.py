import requests
import openai
import json
import re
from dotenv import load_dotenv
import os

load_dotenv()

# === CONFIGURATION ===
openai.api_key = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# === LLM CALL ===
def call_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


# === STEP 1: Extract semantic roles ===
def convert_query_to_wikidata_search(output):
    prompt = f"""
Your job is to extract only the **semantic concepts** from the user's natural language query that should be searched on Wikidata to build a SPARQL query.

- Ignore helper words like "give", "me", "show", "list", etc.
- Focus on real-world **concepts**: things, people, places, classes, etc.
- For each term, assign a semantic role like: "class", "object", "country", "media", "property", etc.

Return as a Python-style list of dictionaries:
[
  {{ "term": "cat", "role": "class" }},
  {{ "term": "image", "role": "media" }}
]

Now extract from this:
"{output}"

Only return the Python list.

Always try to adapt the terms to better fit SPARQL code. Like if the user asks for picture its better to use the term image. Another 
example is if the user asks for musics its better for you to use the term song. So always try to adapt the terms to sparql identifiers
as in those examples.


And also try always to remove plural (albums -> album | dogs -> dog). You just not remove plural if the plural is the name of 
some product (album named Pictures for example)

"""
    text = call_openai(prompt)

    # Clean markdown if needed
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


# === STEP 3: Extract relevant Q-IDs ===
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


# === STEP 4: Retrieve Wikidata labels and descriptions ===
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


# === STEP 5: Generate SPARQL query ===
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
Only use FILTER or regex when really needed.

Now generate a SPARQL query for:
"{nl_query}"

Return only the SPARQL query.
"""
    return call_openai(prompt)


# === MAIN EXECUTION ===
if __name__ == "__main__":
    user_input = input("Enter your query in natural language: ")

    # Step 1: Extract semantic roles
    search_terms_with_roles = convert_query_to_wikidata_search(user_input)
    search_terms_with_roles = normalize_roles(search_terms_with_roles)
    print(f"\n🔍 Extracted terms with roles: {search_terms_with_roles}")

    # Step 2: Extract raw term strings
    term_strings = [entry["term"] for entry in search_terms_with_roles]

    # Step 3: Use Serper search
    search_results_dict = query_search_api(term_strings)

    # Step 4: Get Q-IDs
    wikidata_ids = extract_ids_per_term(search_results_dict)

    # Step 5: Retrieve entity descriptions
    wikidata_entities = get_wikidata_descriptions(wikidata_ids)

    # Step 6: Print entities
    print("\n🔎 Top Wikidata entities:")
    for entity in wikidata_entities:
        print(f"- {entity['label']} ({entity['id']}): {entity['description']}")

    # Step 7: Generate SPARQL
    sparql_query = natural_language_to_sparql(user_input, wikidata_entities, search_terms_with_roles)

    print("\n🧠 Generated SPARQL Query:")
    print(sparql_query)

    # Optional check
    if sparql_query.count("wdt:P31") > 2:
        print("\n⚠️ Note: Query might be overusing `P31 (instance of)`.")
