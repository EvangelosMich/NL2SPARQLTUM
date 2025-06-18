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
from .retriever import retrieve_offline_ids

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

# Load ID mapping for known phrases to Wikidata entity/property IDs





# Load sentence embedding model for semantic similarity
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

# Try loading a spaCy English model
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    print("en_core_web_sm not found, trying en_core_web_trf...")
    try:
        NLP = spacy.load("en_core_web_trf")
    except OSError:
        print("No spaCy model found. Install with: python -m spacy download en_core_web_sm")
        NLP = None

# Disable unused spaCy components for faster inference
if NLP:
    NLP.disable_pipes(["parser", "lemmatizer"])

# Precompute example embeddings for fast retrieval
EXAMPLE_QUESTIONS = [ex["question"] for ex in EXAMPLES]
EXAMPLE_EMBEDDINGS = EMBEDDER.encode(EXAMPLE_QUESTIONS, normalize_embeddings=True)

# ------------------------ Utility Functions ------------------------

    
# Extract named entities using spaCy (or fallback regex)
def extract_named_entities_spacy(text):
    print(text)
    if not NLP:
        # fallback if no spaCy model loaded
        return re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+)*)', text)
    
    doc = NLP(text.strip())
    entities = []

    # Add named entities based on label
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]:
            entities.append(ent.text)

    # Also add capitalized proper nouns not in the entity list
    for token in doc:
        if token.pos_ == "PROPN" and token.text not in entities:
            entities.append(token.text)

    print(entities)
    return list(set(entities))  # return unique entries

# Wrapper function (currently just reuses above)
def extract_named_candidates(text):
    return extract_named_entities_spacy(text)

# Use matcher and fallback to generate ID hints for prompting
def get_id_hints(user_question):

    hints = []

    candidates = extract_named_candidates(user_question)
    candidate_text =  " ".join(candidates)

    entity_ids,prop_ids = retrieve_offline_ids(user_question+ " " + candidate_text)
    for eid in entity_ids:
        hints.append(f"(entity:{eid})")
    for pid in prop_ids:
        hints.append(f"(property:{pid})")

    return " ".join(hints)        


# Retrieve similar example questions from precomputed embeddings
def retrieve_examples(query, top_k=3):
    query_emb = EMBEDDER.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_emb, EXAMPLE_EMBEDDINGS)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [EXAMPLES[i] for i in top_indices]

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
    hints = get_id_hints(user_question)
    print(f"The hints that I've gathered are {hints}")

    retrieved = retrieve_examples(user_question + " " + hints)
    final_prompt = build_prompt(user_question, retrieved, dialog_history, hints)

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT
    )

    response = model.generate_content(final_prompt, generation_config={"temperature": 0.2})
    return response.text

# ------------------------ Main REPL Loop ------------------------

def main():
    dialog_history = []

    while True:
        user_question = input("What would you like to ask?\n> ")
        response = get_llm_response(user_question, dialog_history)
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
