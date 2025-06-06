import os
import json
import requests
import google.generativeai as genai

# Configure the Gemini API
genai.configure(api_key="AIzaSyAVbJ-qe9dZJgOZdR251qIz1aBDHFSyWOw")

# Initialize the Gemini 1.5 Flash model
model = genai.GenerativeModel("gemini-1.5-flash")

def query_search_api(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': 'b4b565c31a58bca9cb7e6201a23a007b3800a537',
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
    return response.json()

def natural_language_to_sparql(nl_query):
    prompt = f"""
    Convert the following natural language question into a SPARQL query:

    "{nl_query}"

    Provide only the SPARQL query without any explanations.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# Main execution
user_input = input("Enter your query in natural language: ")
search_result = query_search_api(user_input)

print("Search Results (titles):")
for i, result in enumerate(search_result.get("organic", [])[:3]):
    print(f"{i+1}. {result.get('title')}")

sparql_query = natural_language_to_sparql(user_input)
print("\nGenerated SPARQL Query:")
print(sparql_query)
