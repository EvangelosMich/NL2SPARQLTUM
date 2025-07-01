# ------------------------ Imports and Environment Setup ------------------------

# Import custom prompt templates
from .prompt_template import SYSTEM_PROMPT, USER_TEMPLATE

# External libraries
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import pandas as pd
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
from spacy.matcher import PhraseMatcher
from difflib import get_close_matches
from .jsonfiles.retriever import retrieve_offline_ids

# Optional CUDA settings for performance (currently forcing CPU usage)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# ------------------------ Configuration and Model Loading ------------------------

# Load API key from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load example question-answer pairs for few-shot prompting
with open(os.path.join(script_dir, '..', 'examples', 'examples.json'), 'r', encoding='utf-8') as f:
    EXAMPLES = json.load(f)


with open(os.path.join(script_dir, '..', 'examples', 'examples_rdfs.json'), 'r', encoding='utf-8') as f:
    EXAMPLES_RDFS = json.load(f)






# Load sentence embedding model for semantic similarity
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')



# Precompute example embeddings for fast retrieval
EXAMPLE_QUESTIONS = [ex["question"] for ex in EXAMPLES]
EXAMPLE_EMBEDDINGS = EMBEDDER.encode(EXAMPLE_QUESTIONS, normalize_embeddings=True)

EXAMPLE_RDFS_QUESTIONS = [ex["question"] for ex in EXAMPLES_RDFS]
EXAMPLE_RDFS_EMBEDDINGS = EMBEDDER.encode(EXAMPLE_RDFS_QUESTIONS, normalize_embeddings=True)

# ------------------------ Utility Functions ------------------------

    
# Extract named entities using spaCy (or fallback regex)
def convert_query_to_wikidata_search(output):
    prompt = f"""
Your job is to extract only the **semantic concepts** from the user's natural language query that should be searched on Wikidata to build a SPARQL query.

- Ignore helper words like "give", "me", "show", "list", etc.
- Focus on real-world **concepts**: things, people, places, classes, etc.

Return as a Python-style list of dictionaries:
[
  {{ "term": "cat"}}
  {{ "term": "image"}}
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
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT
    )
    response = model.generate_content(prompt, generation_config={"temperature": 0.2})

    try:
        print(response.text)
        result = eval(response.text)
        return result
    except Exception:
        return []


# Use matcher and fallback to generate ID hints for prompting
def get_id_hints(user_question):

    hints = []

    candidates = convert_query_to_wikidata_search(user_question)
    candidate_text =  " ".join(candidates)
    candidate_terms = " ".join(c['term'] for c in candidates)

    entity_ids,prop_ids = retrieve_offline_ids(user_question+ " " + candidate_text)
    for eid in entity_ids:
        hints.append(f"(entity:{eid})")
    for pid in prop_ids:
        hints.append(f"(property:{pid})")

    found_entities = bool(entity_ids)    

    return " ".join(hints),found_entities,candidate_terms        


# Retrieve similar example questions from precomputed embeddings
def retrieve_examples(query, top_k=3):
    query_emb = EMBEDDER.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_emb, EXAMPLE_EMBEDDINGS)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [EXAMPLES[i] for i in top_indices]


def retrieve_examples_rdfs(query, top_k=3):
    query_emb = EMBEDDER.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_emb, EXAMPLE_RDFS_EMBEDDINGS)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [EXAMPLES_RDFS[i] for i in top_indices]

# Build the prompt given user query, examples, history, and entity hints
def build_prompt(user_question, examples, dialog_history, hints):
    example_text = "".join([
        f"Question: {ex['question']}\n"
        f"ReasoningStyle: {ex.get('reasoning_style', 'Chain of Thought')}\n"
        f"Thought: {ex['thought']}\n"
        f"SPARQL:\n{ex['sparql']}\n\n"
        for ex in examples
    ])

    # Format the dialogue history
    history_lines = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in dialog_history]
    formatted_history = "\n".join(history_lines)

    user_augmented = f"{user_question}\n{hints}" if hints else user_question

    # Fill the user prompt using a template
    return USER_TEMPLATE.format(
        examples_block=example_text,
        dialog_history=formatted_history,
        user_question=user_augmented,
    )

# Get the final LLM-generated SPARQL using Gemini and prompt
def get_llm_response(user_question, dialog_history):
    hints,found_entities,candidate_terms = get_id_hints(user_question)
    print(hints)
    print(f"The hints that I've gathered are {hints}")

    if not found_entities:
        return "__NO_ENTITY_FOUND__"

    retrieved = retrieve_examples(user_question + " " + hints)
    final_prompt = build_prompt(user_question, retrieved, dialog_history, hints)

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT
    )

    response = model.generate_content(final_prompt, generation_config={"temperature": 0.2})

    return response.text


def build_prompt_rdfs(user_question, examples, dialog_history, candidates):
    example_text = "".join([
        f"Question: {ex['question']}\n"
        f"ReasoningStyle: {ex.get('reasoning_style', 'Chain of Thought')}\n"
        f"Thought: {ex['thought']}\n"
        f"SPARQL:\n{ex['sparql']}\n\n"
        for ex in examples
    ])

    history_lines = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in dialog_history]
    formatted_history = "\n".join(history_lines)

    user_augmented = f"{user_question}\n{candidates}" if candidates else user_question

    return f"""You are a SPARQL expert.

Task:
- Translate the natural-language question to SPARQL.
- Use rdfs:label look-ups (e.g., ?x rdfs:label "Albert Einstein"@en).
- Always add SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} for readable labels.
- Use the correct properties (like wdt:P31) and known constants (like wd:Q5).
- Never guess IDs — use only rdfs:label matches or provided hints.

Use the following examples to guide your approach:

{example_text}

Conversation so far:
{formatted_history}

Now answer the following question using rdfs:label and no QIDs unless explicitly provided:
Question: {user_augmented}
Your answer should follow this structure:
Thought: <reasoning>
SPARQL:
<query>"""

def get_llm_response_rdfs(user_question,dialog_history):
      # Step 1: Extract candidate labels from the question
    candidates = convert_query_to_wikidata_search(user_question)
    label_hints = [f'(label:"{entry["term"]}"@en)' for entry in candidates]
    hint_text = " ".join(label_hints)

    # Step 2: Retrieve relevant examples as usual
    retrieved = retrieve_examples_rdfs(user_question)
    print(retrieved)

    # Step 3: Build the prompt with label-based hints
    prompt = build_prompt_rdfs(user_question, retrieved, dialog_history, hint_text)

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT
    )

    response = model.generate_content(prompt, generation_config={"temperature": 0.2})
    print("RDFS RETRY activated with question:", user_question)
    print(response.text)
    return response.text


# ------------------------ Main REPL Loop ------------------------

def main():
    dialog_history = []

    while True:
        user_question = input("What would you like to ask?\n> ")
        response = get_llm_response_rdfs(user_question,dialog_history)
        print("\n" + response + "\n")

        # Save history for context-aware prompting
        dialog_history.append({"role": "user", "content": user_question})
        dialog_history.append({"role": "assistant", "content": response})

        satisfied = input("Are you satisfied with this result? (yes/no)\n> ")
        if satisfied.lower() == "yes":
            # TODO: Optionally log the example
            break

# Run when called directly
if __name__ == "__main__":
    main()
