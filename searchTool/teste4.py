import requests
import openai
import json
import re

# Configure OpenAI API
openai.api_key = (
    "sk-proj-kCLczx7_1zyWErWOqCnavSTcYpjt5bhizlzjey4qTcJlk3Mljr-"
    "zBwmpbR0kwqbMZXr9ccS33ST3BlbkFJ7gril2RheKhSuhh3qBET0GKYb4kKwBEbqM85dl2"
    "FtrmilTw0iChuI5B96BVxX25tvg3ZkIRREA"
)


def call_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def convert_query_to_wikidata_search(output):
    prompt = f"""
Your job is to extract only the **semantic concepts** from the user's natural language query that should be searched on Wikidata to build a SPARQL query.

- Ignore helper words like "give", "me", "show", "list", "tell", "get", etc.
- Focus on real-world **concepts**: things, people, places, classes, media types, properties, etc.
- For each term, assign a semantic role like: "class", "object", "country", "media", "property", "color", etc.

Return the result as a Python-style list of dictionaries:
[
  {{ "term": "cat", "role": "class" }},
  {{ "term": "image", "role": "media" }}
]

Examples:
Input: "give me cat pictures"
Return:
[
  {{ "term": "cat", "role": "class" }},
  {{ "term": "image", "role": "media" }}
]

Input: "show me Italian directors"
Return:
[
  {{ "term": "Italy", "role": "country" }},
  {{ "term": "director", "role": "class" }}
]

Now extract the semantic terms from this query:
"{output}"

Only return the Python list.
"""
    text = call_openai(prompt)

    # Remove Markdown formatting if present
    if text.startswith("```"):
        text = text.strip("```").strip()
        if text.startswith("python"):
            text = text[len("python"):].strip()

    terms = eval(text)

    # Validate
    if isinstance(terms, list) and all("term" in t and "role" in t for t in terms):
        return terms
    else:
        raise ValueError("LLM output did not follow expected structure")




def query_search_api(terms):
    url = "https://google.serper.dev/search"
    headers = {
        'X-API-KEY': 'b4b565c31a58bca9cb7e6201a23a007b3800a537',
        'Content-Type': 'application/json'
    }

    term_results = {}
    for term in terms:
        payload = json.dumps({"q": term + " site:wikidata.org"})
        response = requests.post(url, headers=headers, data=payload)
        data = response.json()
        term_results[term] = data.get("organic", [])
    
    return term_results


def extract_ids_per_term(results_dict, max_ids_per_term=4):
    all_ids = []
    seen = set()

    for term, results in results_dict.items():
        count = 0
        for result in results:
            link = result.get("link", "")
            match = re.search(r"wikidata\.org/wiki/(Q\d+)", link)
            if match:
                qid = match.group(1)
                if qid not in seen:
                    seen.add(qid)
                    all_ids.append(qid)
                    count += 1
                if count == max_ids_per_term:
                    break
    return all_ids


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


def natural_language_to_sparql(nl_query, entities, role_map):
    context_lines = [f"{e['label']} ({e['id']}): {e['description']}" for e in entities]
    context = "Here are some candidate entities from Wikidata:\n" + "\n".join(context_lines)

    role_instructions = {
        "color": "Use property P462 (color)",
        "class": "Use property P31 (instance of), but prefer specific properties when possible",
        "occupation": "Use property P106 (occupation)",
        "country": "Use P495 (country of origin) or P27 (citizenship)",
        "object": "Use contextually appropriate properties like P57 (director), P50 (author)",
        "media": "Use P18 (image), P1651 (YouTube ID), etc.",
        "human": "Use P31 wd:Q5 for humans, but combine with P106 (occupation)",
    }

    mapped_roles = "\n".join(
        f"{r['term']} → {r['role']}: {role_instructions.get(r['role'], 'Use best-fit property')}"
        for r in role_map
    )

    prompt = f"""
You are converting a natural language question into a SPARQL query for Wikidata.

Here are going to be options that maybe do not correspond to the query. Pick the right ones to transform into SPARQL. Always
pay attention to details like nationality and color and everything else in this regard.

PAY MUCH ATTENTION TO LABELS AS MOST OF THE TIME THEY GIVE HINTS BASED ON WHAT YOU NEED TO USE ACTUALLY

for example here: female (Q6581072): to be used in "sex or gender" (P21) to indicate that the human subject is a female or "semantic gender" (P10339) to indicate that a word refers to a female person

if you need to use human female in this case you would use P21 not Q6581072

{context}

The following terms and their semantic roles were extracted from the question:
{mapped_roles}

Based on these, write a SPARQL query using the most appropriate Wikidata properties.
Avoid overusing P31 unless the item is truly a class/entity type.
Use specific properties like P106 (occupation) or P495 (origin) when applicable.

Remember to be sure about the carachteristics, like nationality and color and everything in that regard

Now generate a SPARQL query for:
"{nl_query}"

Return only the SPARQL query.
"""
    return call_openai(prompt)



# === MAIN EXECUTION ===

user_input = input("Enter your query in natural language: ")

# Step 1: Extract semantic search terms
search_terms_with_roles = convert_query_to_wikidata_search(user_input)
print(f"\n🔍 Extracted terms with roles: {search_terms_with_roles}")

# Step 2: Get just the term strings
term_strings = [entry["term"] for entry in search_terms_with_roles]

# Step 3: Query Serper
search_results_dict = query_search_api(term_strings)

# Step 4: Extract Wikidata Q-IDs
wikidata_ids = extract_ids_per_term(search_results_dict)

# Step 5: Get label/description
wikidata_entities = get_wikidata_descriptions(wikidata_ids)

# Step 6: Print found entities
print("\n🔎 Top Wikidata entities:")
for entity in wikidata_entities:
    print(f"- {entity['label']} ({entity['id']}): {entity['description']}")

# Step 7: Ask OpenAI for SPARQL generation
sparql_query = natural_language_to_sparql(user_input, wikidata_entities, search_terms_with_roles)

print("\n🧠 Generated SPARQL Query:")
print(sparql_query)
