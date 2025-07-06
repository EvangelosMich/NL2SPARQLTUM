import streamlit as st
import pandas as pd
import RAGModel.llmbasedbackend as lm
import json
import numpy as np
import captureSparql as cs
import fastapi
import requests
import os
import time
import searchTool.searchtool as sa

# Ignore torch file watcher issue
os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch"

# Custom font size for radio buttons
st.markdown("""
    <style>
    .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {
        font-family: Arial, sans-serif;
        font-size: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# Choose backend
backend_choice = st.radio("Choose backend implementation", ("LLM Based Agent", "RAG Model"))

# Title
st.title(f"NL TO SPARQL LLM ({backend_choice})")

# SPARQL endpoint
url = 'https://query.wikidata.org/sparql'

# Init session state
if "dialog_history" not in st.session_state:
    st.session_state.dialog_history = []

if "last_user_question" not in st.session_state:
    st.session_state.last_user_question = ""

if "show_retry_button" not in st.session_state:
    st.session_state.show_retry_button = False

if "retry_rdfs" not in st.session_state:
    st.session_state.retry_rdfs = False

# Display previous messages
for message in st.session_state.dialog_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "table" in message:
            df = pd.DataFrame(message["table"])
            st.table(df)
        if "time" in message:
            st.write(f"⏱️ Answered in {round(message['time'], 2)} seconds")

# -------- Helper: Typewriter animation --------
def typewriter(text: str, speed: int):
    tokens = text.split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        container.markdown(" ".join(tokens[:index]))
        time.sleep(1 / speed)

# -------- Helper: Execute SPARQL query --------
def execute_sparql_query(sparql_code, show_results=True):
    """Execute SPARQL query and return results"""
    query = cs.clean_query(sparql_code)
    if not query:
        if show_results:
            st.error("❌ No valid SPARQL query extracted.")
        return None, None
    
    r = requests.get(url, params={'format': 'json', 'query': query},
                     headers={"Accept": "application/sparql-results+json"})
    if not r.ok:
        if show_results:
            st.error(f"❌ SPARQL endpoint returned an error: {r.status_code}")
            st.text(r.text)
        return None, None
    
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        if show_results:
            st.error(f"❌ Failed to parse JSON: {str(e)}")
            st.text(r.text)
        return None, None
    
    bindings = data.get("results", {}).get("bindings", [])
    flat_rows = [{k: v["value"] for k, v in row.items()} for row in bindings]
    df = pd.DataFrame(flat_rows)
    
    return df, query

# -------- Helper: Display query results --------
def display_query_results(df, elapsed_time):
    """Display query results in chat"""
    with st.chat_message("assistant"):
        st.table(df)
        st.write(f"Number of results: {len(df)}")
        st.write(f"Question was answered in {round(elapsed_time, 2)} seconds")

# -------- Helper: Process user question --------
def process_user_question(user_quest, use_rdfs=False):
    """Process user question and return answer"""
    start_time = time.time()
    
    # Add user message to history
    st.session_state.dialog_history.append({"role": "user", "content": user_quest})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_quest)
    
    # Get answer based on backend choice or RDFS retry
    if use_rdfs:
        answer = lm.get_llm_response_rdfs(user_quest, st.session_state.dialog_history)
    elif backend_choice == "RAG Model":
        answer = lm.get_llm_response(user_quest, st.session_state.dialog_history)
    else:
        search_terms_with_roles = sa.convert_query_to_wikidata_search(user_quest)
        search_terms_with_roles = sa.normalize_roles(search_terms_with_roles)
        term_strings = [entry["term"] for entry in search_terms_with_roles]
        search_results_dict = sa.query_search_api(term_strings)
        wikidata_ids = sa.extract_ids_per_term(search_results_dict)
        wikidata_entities = sa.get_wikidata_descriptions(wikidata_ids)
        answer = sa.natural_language_to_sparql(user_quest, wikidata_entities, search_terms_with_roles)
        print(answer)
    
    # Display answer
    with st.chat_message("assistant"):
        if "SPARQL:" in answer:
            reasoning, sparql_code = answer.split("SPARQL:", 1)
            typewriter(reasoning.strip(), 10)
            st.code(sparql_code.strip(), language="sparql")
        else:
            sparql_code = answer.strip()
            st.code(sparql_code, language="sparql")
    
    # Execute query and get results
    df, query = execute_sparql_query(sparql_code)
    if df is None:
        return
    
    # Display results
    elapsed_time = time.time() - start_time
    display_query_results(df, elapsed_time)
    
    # Add assistant message to history
    st.session_state.dialog_history.append({
        "role": "assistant",
        "content": answer,
        "table": df.to_dict(),
        "time": elapsed_time
    })
    
    # Auto scroll
    st.components.v1.html("""
    <script>
        function scrollToBottom() {
            let container = parent.document.querySelector('section.main');
            if (!container) return;

            let lastHeight = -1;
            let attempts = 0;

            function tryScroll() {
                const currentHeight = container.scrollHeight;
                if (currentHeight !== lastHeight) {
                    container.scrollTop = currentHeight;
                    lastHeight = currentHeight;
                    attempts = 0;
                    setTimeout(tryScroll, 200);
                } else if (attempts < 5) {
                    attempts++;
                    setTimeout(tryScroll, 200);
                }
            }

            tryScroll();
        }

        scrollToBottom();
    </script>
    """, height=0)
    
    # Show retry button if no results and not already using RDFS

    st.session_state.show_retry_button = True
    st.session_state.last_user_question = user_quest


# -------- Main logic --------

# Handle RDFS retry first (if triggered)
if st.session_state.retry_rdfs:
    process_user_question(st.session_state.last_user_question, use_rdfs=True)
    st.session_state.retry_rdfs = False

# Chat input
user_quest = st.chat_input("How can I assist you?")

# Process new user input
if user_quest:
    st.session_state.last_user_question = user_quest
    process_user_question(user_quest)

# Show retry button after results are displayed (if needed)
if st.session_state.show_retry_button:
    st.warning("Do you think the results we provided you with were false? Would you like to try again with our RDFS method?")
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        pass
    with col2:pass
    with col4:pass
    with col5:pass
    with col3:
     if st.button("🔄 Yes, retry"):
        st.session_state.retry_rdfs = True
        st.session_state.show_retry_button = True
        st.rerun()