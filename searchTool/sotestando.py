import requests
import google.generativeai as genai
import json
import re

# Configure the Gemini API
genai.configure(api_key="AIzaSyAVbJ-qe9dZJgOZdR251qIz1aBDHFSyWOw")
model = genai.GenerativeModel("gemini-2.0-flash")

def query_search_api(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query + " site:wikidata.org"})
    headers = {
        'X-API-KEY': 'b4b565c31a58bca9cb7e6201a23a007b3800a537',
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
    return response.json()

def extract_wikidata_ids(results, max_ids=4):
    ids = []
    for result in results.get("organic", []):
        link = result.get("link", "")
        match = re.search(r"wikidata\.org/wiki/(Q\d+)", link)
        if match:
            qid = match.group(1)
            if qid not in ids:
                ids.append(qid)
            if len(ids) == max_ids:
                break
    return ids

def get_wikidata_descriptions(qids, language='en'):
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

# === Example usage ===


def convert_query_to_wikidata_search(output):
    
  
    prompt = f"""
The following input is a natural language query that the user wants to transform into SPARQL code. But to do that we need to 
find the entity ids in the wikidata. I have a function for that already, but i need that you give me only the object needed 
from this query so i can search with my function. 

Example of NL query: What is the director of lord of the rings movie?

You should give me only the words: Lord of The Rings

is the object i want to search not the answer to the question!!!

and try to aim for the most unknown object of 


"{output}"


Return ONLY the entity i need to search in the wikidata endpoint!
"""
    response = model.generate_content(prompt)
    return response.text.strip()


def natural_language_to_sparql(nl_query, entity_id=None):
    context = f"The entity is {entity_id}." if entity_id else ""
    prompt = f"""
Convert the following natural language question into a SPARQL query for the Wikidata endpoint.

the context should be a list of ids of the first 4 results regarding the focus of the query. considering the user input and the wikidata
results, pick the most fitting id

USE THE CONTEXT TO KNOW THE IDS NECESSARY

Natural language question:
"{nl_query}"

{context}

Return only the SPARQL query.
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# Main execution
user_input = input("Enter your query in natural language: ")

objeto = convert_query_to_wikidata_search(user_input)
print(f"\n Aqui o teste porra: {objeto}")

ids_plus_descrpt1 = query_search_api(objeto)
ids_plus_descrpt2 = extract_wikidata_ids(ids_plus_descrpt1)
ids_plus_descrpt3 = get_wikidata_descriptions(ids_plus_descrpt2)

for entity in ids_plus_descrpt3:
    print(f"{entity['label']} ({entity['id']}): {entity['description']}")

#entity_id = search_wikidata_entity(teste)
print(f"\n🔍 Top Wikidata Entity ID: {ids_plus_descrpt3}")

sparql_query = natural_language_to_sparql(user_input, ids_plus_descrpt3)
print("\n🧠 Generated SPARQL Query:")
print(sparql_query)
