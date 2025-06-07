import requests
import RAGModelHenrique.llmbasedbackend as askai
from fastapi import FastAPI
import json
import re

# print(repr(askai.result))
url = 'https://query.wikidata.org/sparql'
# response = requests.post("https://query.wikidata.org/sparql",
#                          data = {"query" : askai.result},
#                          headers = {"Content-Type": "application/x-www-form-urlencoded",
#                                     "Accept": "application/sparql-results+json"})
#userquestion = input("What do I show you\n")
#dialog_history = [f"User:{userquestion}"]


def extract_sparql_from_response(response_text):
    # Match common SPARQL query types with optional prefixes and whitespace
    match = re.search(r"(?i)(SELECT|ASK|CONSTRUCT|DESCRIBE)\s+.*", response_text, re.DOTALL)
    if match:
        # This grabs everything starting from the SPARQL keyword
        return response_text[match.start():].strip()
    else:
        return None

#test = askai.get_llm_response(userquestion,dialog_history)
#query = extract_sparql_from_response(test)
'''
if query is None:
    print("❌ No valid SPARQL query found in the LLM response.")
    exit()

print(query)
'''
def clean_query(query):
    # Remove markdown backticks or language hints
    query = query.strip()
    query = re.sub(r"^```sparql\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"^```|```$", "", query)
    return query.strip()
'''
r = requests.get(url, params = {'format': 'json', 'query': query})
data = r.json()
bindings = data.get("results",{}).get("bindings",[])
if not bindings:
    print(askai.get_llm_response(f"The following SPARQL query was generated for the question: {askai.user_question}.It returned no results. Can you identify the mistake and generate a corrected version?"))

print(data)    '''