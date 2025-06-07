from .prompt_template import SYSTEM_PROMPT,USER_TEMPLATE
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

#Load API keys and enviroment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


#Load examples
with open(os.path.join(script_dir,'..','examples','examples.json'),'r',encoding='utf-8') as f:
    EXAMPLES =  json.load(f)


#Load Id-Mapping
with open(os.path.join(script_dir,'..','examples','example_id.json'),'r',encoding='utf-8') as f:
    ID_MAP = json.load(f)

#Embedding model from model.py

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
EXAMPLE_QUESTIONS = [ex["question"] for ex in EXAMPLES]
EXAMPLE_EMBEDDIGNS = EMBEDDER.encode(EXAMPLE_QUESTIONS,normalize_embeddings=True)

def resolve_id(phrase):
    if phrase in ID_MAP:
        return ID_MAP[phrase]
    #fallback to wikidata if id not found
    response = requests.get("https://www.wikidata.org/w/api.php", params= {
        "action" :"wbsearchentities",
        "search" : phrase,
        "language": "en",
        "format":"json",
        })       
    
    result = response.json().get("search",[])
    if result:
     entity_id = result[0]["id"]
     if entity_id.startswith("Q"):
         return "wd:"+ entity_id
     elif entity_id.startswith("P"):
         return "wdt:" + entity_id
     else:
         return None


def extract_named_candidates(text):
    # Basic heuristic: extract proper-noun phrases (capitalized sequences)
        return re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+)*)', text)

def get_id_hints(user_question):


    hints = []
    lower_quest = user_question.lower()
    for phrase,wikidata_path in ID_MAP.items():
        
        if phrase in lower_quest:
            hints.append(f"(hint:{phrase} = {wikidata_path})")
    if not hints:
        candidates = extract_named_candidates(user_question)
        print(candidates)
        for candidate in candidates:
          fallback = resolve_id(candidate)
          if fallback:
            hints.append(f"(fallback:{user_question} = {fallback})")

    return " ".join(hints)

def retrieve_examples(query,top_k=3):
    query_emb = EMBEDDER.encode([query],normalize_embeddings=True)
    scores = cosine_similarity(query_emb,EXAMPLE_EMBEDDIGNS)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [EXAMPLES[i] for i in top_indices]


def build_prompt(user_question,examples,dialog_history,hints):
    example_text = "".join([
    f"Question: {ex['question']}\n"
    f"ReasoningStyle: {ex.get('reasoning_style', 'Chain of Thought')}\n"
    f"Thought: {ex['thought']}\n"
    f"SPARQL:\n{ex['sparql']}\n\n"
    for ex in examples
            ])

    history_lines = [f"{msg['role'].capitalize()}:{msg['content']}" for msg in dialog_history]      
    formatted_history = "\n".join(history_lines)

    user_augmented = f"{user_question}\n{hints}" if hints else user_question
    
    
    return USER_TEMPLATE.format(
        examples_block = example_text,
        dialog_history = formatted_history,
        user_question = user_augmented,
    )        


def get_llm_response(user_question,dialog_history):
# get hints in case there are any
    hints = get_id_hints(user_question)
    print(f"The hits that ive gathered are{hints}")
    retrieved = retrieve_examples(user_question+" "+hints)
    final_prompt = build_prompt(user_question,retrieved,dialog_history,hints)

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=  SYSTEM_PROMPT
    )

    response = model.generate_content(final_prompt,generation_config={"temperature":0.2})
    return response.text




def main():
    dialog_history = []

    while True:
        user_question = input("What would you like to ask?\n> ")
        response = get_llm_response(user_question, dialog_history)
        print("\n" + response + "\n")

        dialog_history.append({"role": "user", "content": user_question})
        dialog_history.append({"role": "assistant", "content": response})

        satisfied = input("Are you satisfied with this result? (yes/no)\n> ")
        if satisfied.lower() == "yes":
            #log_example_if_confirmed(user_question, response)
            break
        else:
            continue

if __name__ == "__main__":
    main()
    