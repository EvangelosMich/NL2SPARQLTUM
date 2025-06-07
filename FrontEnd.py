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



os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch"

implementations = ["RAG_Model","LLM-Based-Agent"]
impl_input = st.selectbox('Choose the desired Implementation', implementations)


def typewriter(text: str, speed: int):
    tokens = text.split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text)
        time.sleep(1 / speed)

st.title("NL TO SPARQL LLM")

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

if user_quest:
    with st.chat_message("user"):
       st.markdown(user_quest)
    answer = lm.get_llm_response(user_quest, st.session_state.dialog_history)
    st.session_state.dialog_history.append({"role":"user","content":user_quest})
    with st.chat_message("assistant"):
     if "SPARQL:" in answer:
        reasoning_part, sparql_code = answer.split("SPARQL:", 1)
        typewriter(reasoning_part.strip(), 10)
        st.code(sparql_code.strip(), language="sparql")
     else:
        # fallback if split fails
        typewriter(answer, 10)
       

    query = cs.extract_sparql_from_response(answer)
    query = cs.clean_query(query)
    st.markdown(query)
    if not query:
     st.error("❌ No valid SPARQL query extracted from LLM response.")
     st.stop()

    headers = {
    "Accept": "application/sparql-results+json"
}
    r = requests.get(url,params={'format': 'json', 'query': query},headers=headers)
    data = r.json()
    bindings = data.get("results",{}).get("bindings",[])
    flat_rows = [{k:v["value"] for k,v in row.items()} for row in bindings]
    
    df = pd.DataFrame(flat_rows)
    with st.chat_message("assistant"):
        st.table(df)
        st.write(f"Number of results: {len(df)}")

    st.session_state.dialog_history.append({
    "role": "assistant",
    "content": answer,
    "table": df.to_dict()  # or json
})   
