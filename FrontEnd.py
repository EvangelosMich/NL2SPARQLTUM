import streamlit as st
import pandas as pd
import RAGModelHenrique.llmbasedbackend as lm
import json
import numpy as np
import captureSparql as cs
import fastapi
import requests
import os
import time
import searchTool.sotestando as sa



os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch"
import streamlit as st

# Add custom CSS to change the font
st.markdown(
    """
    <style>
    .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {
        font-family: Arial, sans-serif; /* Example font family */
        font-size: 30px; /* Example font size */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
backend_choice = st.radio("Choose backend implementation",("LLM Based Agent","RAG Model"))


def typewriter(text: str, speed: int):
    tokens = text.split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text)
        time.sleep(1 / speed)
if backend_choice == "LLM Based Agent":
    st.title("NL TO SPARQL LLM (LLM Based Agent)")
else:
   st.title("NL TO SPARQL LLM (RAG Model)")
with st.chat_message("assistant"):
 user_quest = st.chat_input("How can I assist you?")

 
url = 'https://query.wikidata.org/sparql'

if "dialog_history" not in st.session_state:
   st.session_state.dialog_history = []

for message in st.session_state.dialog_history:
   with st.chat_message(message["role"]):
      st.markdown(message["content"])
      if "table" in message:
        df = pd.DataFrame(message["table"])
        st.table(df)
      if "time" in message:
            st.write(f"⏱️ Answered in {round(message['time'], 2)} seconds")     

if user_quest:
    start_time = time.time()
    with st.chat_message("user"):
       st.markdown(user_quest)
    if backend_choice == "RAG Model":   
     answer = lm.get_llm_response(user_quest, st.session_state.dialog_history)
    else:
       answer =  sa.natural_language_to_sparql(user_quest)
       print(answer)
    st.session_state.dialog_history.append({"role":"user","content":user_quest})
    with st.chat_message("assistant"):
     sparql_code = None
     if "SPARQL:" in answer:
        reasoning_part, sparql_code = answer.split("SPARQL:", 1)
        typewriter(reasoning_part.strip(), 10)
        st.code(sparql_code.strip(), language="sparql")
     else:
        # fallback if split fails
        sparql_code = answer.strip()
        st.code(sparql_code, language="sparql")

    
    query = cs.clean_query(sparql_code)
    if not query:
     st.error("❌ No valid SPARQL query extracted from LLM response.")
     st.stop()

    headers = {
    "Accept": "application/sparql-results+json"
}
    r = requests.get(url,params={'format': 'json', 'query': query},headers=headers)
    if not r.ok:
        st.error(f"❌ SPARQL endpoint returned an error: {r.status_code}")
        st.text(r.text)  # this will likely show an HTML error page or details
        st.stop()

    try:
        data = r.json()
    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse JSON: {str(e)}")
        st.text(r.text)  # show raw response to debug
        st.stop()
    data = r.json()
    bindings = data.get("results",{}).get("bindings",[])
    flat_rows = [{k:v["value"] for k,v in row.items()} for row in bindings]
    
    df = pd.DataFrame(flat_rows)
    with st.chat_message("assistant"):
        st.table(df)
        st.write(f"Number of results: {len(df)}")
        end_time = time.time()
        elapsed_time = end_time-start_time
        st.write(f"Question was answered in {round(elapsed_time,2)} seconds")

    st.session_state.dialog_history.append({
    "role": "assistant",
    "content": answer,
    "table": df.to_dict(),
    "time": elapsed_time  # or json
})   
