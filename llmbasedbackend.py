import openai
from prompt_template import SYSTEM_PROMPT,USER_TEMPLATE
import google.generativeai as genai
import os
from dotenv import load_dotenv
from retriever import example_retriever
import json
import pandas as pd
import nltk


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
from prompt_template import SYSTEM_PROMPT, USER_TEMPLATE

# def preprocessUserQuestion(user_question):
#    user_question = user_question.lower()
#    Entity_cache = "entity_phrase.json"
#    hints = []
#    with open(Entity_cache,'r',encoding="utf-8") as f:
#         phrase_data = json.load(f)

#    for category in phrase_data:
#     for phrase in phrase_data[category]:
#        if phrase in user_question:
#           response = er.wikidata_search(phrase)
#           if not response:
#              continue
#           response_data = response.json()
#           if "search" in response_data and len(response_data["search"]) >0:
             
#            qid = response_data['search'][0]['id']
#            hints.append(f"(hint:{phrase} = wd:{qid})")
#    if hints:
#       user_question+= " "+ " ".join(hints)         
#    return user_question      
   

#user_question = input("What would you like me to show you?\n")
# def extracting_key(user_question):
#     extraction_prompt = f"""
# You are an intelligent assistant. Given a user's question, extract the most relevant named entity (such as a person, country, place, or political position) in a form that will match labels in Wikidata search. Avoid plural and ambiguous terms. Return the best singular, Wikidata-friendly phrase.

# Only return the phrase — no explanation.

# Question: "{user_question}"
# Keyword:
# """
    
#     response = openai.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are an entity linker for Wikidata."},
#             {"role": "user", "content": extraction_prompt}
#         ],
#         temperature=0.0
#     )

#     keyword = response.choices[0].message.content.strip()
#     return keyword

def synonym_extractor(phrase):
     from nltk.corpus import wordnet
     synonyms = []

     for syn in wordnet.synsets(phrase):
          for l in syn.lemmas():
               synonyms.append(l.name())
     return synonyms          

def get_id_hints(user_question,id_map_path = "examples/example_id.json"):
    with open(id_map_path,'r',encoding='utf-8') as f:
        id_map = json.load(f)

    hints = []
    lower_quest = user_question.lower()
    for phrase,wikidata_path in id_map.items():
        synonyms = synonym_extractor(phrase)
        if phrase in lower_quest or any(syn in lower_quest for syn in synonyms):
            hints.append(f"(hint:{phrase} = {wikidata_path})")
    return " ".join(hints)             

def get_llm_response(user_question,dialog_history):
# get hints in case there are any
    hints = get_id_hints(user_question)
    user_question_augmented = user_question+""+hints

# 1. Retrieve similar examples
    retriever = example_retriever()
    retrieved = retriever.retrieve(user_question_augmented)

# 2. Format them for the prompt
    example_text = ""
    for ex in retrieved:
        example_text += (
    f"Question: {ex['question']}\n"
    f"ReasoningStyle: {ex.get('reasoning_style', 'Chain of Thought')}\n"
    f"Thought: {ex['thought']}\n"
    f"SPARQL:\n{ex['sparql']}\n\n"
)    
    history_lines = []
    for msg in dialog_history:
        role = msg["role"].capitalize()
        content = msg["content"]
        history_lines.append(f"{role}: {content}")
    formatted_history = "\n".join(history_lines)

    # 3. Construct final prompt
    current_prompt = USER_TEMPLATE.format(
        examples_block=example_text,
        dialog_history=formatted_history,
        user_question = user_question_augmented
    )

    # 4. Use with Gemini or OpenAI
   

    model = genai.GenerativeModel(
       model_name="gemini-1.5-flash",
       system_instruction= SYSTEM_PROMPT
    )

    response = model.generate_content(current_prompt,generation_config={"temperature":0.3})
    return response.text



def main():
 print("Now Im ready")   
 user_question = input("What would you like me to show you?\n")




 dialog_history = []

 while True:
    response_text = get_llm_response(user_question, dialog_history)
    print("\n" + response_text + "\n")
    dialog_history.append(f"User: {user_question}")
    dialog_history.append(f"Assistant: {response_text}")

    satisfied = input("Are you satisfied with this result? (yes/no)\n> ")
    if satisfied.lower() == "yes":
        try:
            # Example parsing logic – you can refine this
            parts = response_text.split("SPARQL:")
            thought = parts[0].replace("Thought:", "").strip()
            sparql = parts[1].strip()

            example_entry = {
                "question": user_question,
                "reasoning_style": "Chain of Thought",  # or detect based on content
                "thought": thought,
                "sparql": sparql
            }

            # Load existing examples
            with open("examples/examples.json", "r", encoding="utf-8") as file:
                data = json.load(file)

            data.append(example_entry)

            # Save back to examples.json
            with open("examples/examples.json", "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)

            print("✅ Example successfully added to examples.json")

        except Exception as e:
            print("❌ Failed to extract and save example:", e)

        break
    else:
        followup = input("What would you like to change or ask next?\n> ")
        user_question = followup
        dialog_history.append(f"User: {followup}")

if __name__ == "__main__":
    main()
    