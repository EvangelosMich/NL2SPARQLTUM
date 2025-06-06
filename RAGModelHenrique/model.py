import requests
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Configuration ---
genai.configure(api_key="AIzaSyAVbJ-qe9dZJgOZdR251qIz1aBDHFSyWOw")

# --- FAISS Setup ---
example_questions = [
    "Who directed Inception?",
    "When was Titanic released?",
    "Who played Harry Potter?",
    "What is the capital of France?"
]

example_sparqls = [
    "SELECT ?director WHERE { wd:Q43361 wdt:P57 ?director . }",
    "SELECT ?date WHERE { wd:Q44578 wdt:P577 ?date . }",
    "SELECT ?actor WHERE { wd:Q8337 wdt:P161 ?actor . }",
    "SELECT ?capital WHERE { wd:Q142 wdt:P36 ?capital . }"
]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
question_vectors = embedding_model.encode(example_questions, convert_to_numpy=True)

dimension = question_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_vectors)

# --- Retrieval Function ---
def retrieve_similar_questions(query, k=2):
    query_vector = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, k)
    return [(example_questions[i], example_sparqls[i]) for i in indices[0]]

# --- Entity Retrieval (Films Only) ---
def get_film_entity_id(label):
    query = f'''
    SELECT ?item WHERE {{
      ?item rdfs:label "{label}"@en .
      ?item wdt:P31 wd:Q11424 .
    }} LIMIT 1
    '''
    response = requests.get(
        "https://query.wikidata.org/sparql",
        headers={"Accept": "application/json"},
        params={"query": query}
    ).json()
    results = response["results"]["bindings"]
    return "wd:" + results[0]["item"]["value"].split("/")[-1] if results else None

# --- Property Retrieval by Label ---
def get_property_id(label):
    query = f'''
    SELECT ?property WHERE {{
      ?property a wikibase:Property ;
                rdfs:label "{label}"@en .
    }} LIMIT 1
    '''
    response = requests.get(
        "https://query.wikidata.org/sparql",
        headers={"Accept": "application/json"},
        params={"query": query}
    ).json()
    results = response["results"]["bindings"]
    return "wdt:" + results[0]["property"]["value"].split("/")[-1] if results else None

# --- Prompt Construction ---
def build_prompt(question, entity_label, property_label, entity_id, property_id, retrieved_examples=[]):
    examples_text = "\n".join([
        f"Q: {q}\nA: {sparql}" for q, sparql in retrieved_examples
    ])
    return f"""You are a SPARQL expert. Given an English question and some example queries, generate a correct SPARQL query.

Examples:
{examples_text}

Now, generate SPARQL for:
Entities:
- {entity_label} = {entity_id}
Properties:
- {property_label} = {property_id}

Question: {question}

SPARQL:
"""

# --- Gemini SPARQL Generator ---
def generate_sparql_from_gemini(prompt):
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content([prompt])
    return response.text.strip()

# --- Demo Function ---
def rag_sparql_demo(question, entity_label, property_label):
    entity_id = get_film_entity_id(entity_label)
    property_id = get_property_id(property_label)

    if not entity_id or not property_id:
        return {"error": "Entity or property not found."}

    retrieved = retrieve_similar_questions(question, k=2)
    prompt = build_prompt(question, entity_label, property_label, entity_id, property_id, retrieved)
    sparql = generate_sparql_from_gemini(prompt)

    return {
        "prompt": prompt,
        "generated_query": sparql
    }

# --- Example Usage ---
result = rag_sparql_demo(
    question="Who directed Harry Potter and the Philosopher's Stone?",
    entity_label="Harry Potter and the Philosopher's Stone",
    property_label="director"
)

print(result["prompt"])
print(result["generated_query"])
