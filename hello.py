import streamlit as st
import pandas as pd
import llmbasedbackend as lm
import json
import numpy as np
import captureSparql as cs
import fastapi
import requests
import os


os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch"

st.markdown(
    """
    <style>
    .user_message {
        background-color: #A8DADC;
        color: #1D3557;
        padding: 10px;
        border-radius: 20px;
        margin: 10px 0;
        width: fit-content;
        max-width: 70%;
        align-self: flex-end;
    }
    .bot_message {
        background-color: #F1FAEE;
        color: #457B9D;
        padding: 10px;
        border-radius: 20px;
        margin: 10px 0;
        width: fit-content;
        max-width: 70%;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    </style>
    """,
    unsafe_allow_html=True
)
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
       st.markdown(answer)

    query = cs.extract_sparql_from_response(answer)
    query = cs.clean_query(query)
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
