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

os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch"

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

url = 'https://query.wikidata.org/sparql'

# Initialize session state variables
if "dialog_history" not in st.session_state:
    st.session_state.dialog_history = []

if "retry_with_label" not in st.session_state:
    st.session_state.retry_with_label = False

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "waiting_for_retry_decision" not in st.session_state:
    st.session_state.waiting_for_retry_decision = False

if "retry_decision_made" not in st.session_state:
    st.session_state.retry_decision_made = False

if "last_sparql_code" not in st.session_state:
    st.session_state.last_sparql_code = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

# Display dialog history
for message in st.session_state.dialog_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sparql_code" in message and message["sparql_code"]:
            st.code(message["sparql_code"], language="sparql")
        if "table" in message:
            df = pd.DataFrame(message["table"])
            st.table(df)
        if "time" in message:
            st.write(f"⏱️ Answered in {round(message['time'], 2)} seconds")

# Handle different states
backend_used = None
user_quest = None

# Check if we're waiting for a retry decision
if st.session_state.waiting_for_retry_decision and not st.session_state.retry_decision_made:
    # Display the last SPARQL code that produced 0 results
    if st.session_state.last_sparql_code:
        with st.chat_message("assistant"):
            st.markdown("**SPARQL query that returned 0 results:**")
            st.code(st.session_state.last_sparql_code, language="sparql")
            # Show the empty results table
            st.table(pd.DataFrame())
            st.write("Number of results: 0")
    
    st.warning("It appears the latest query delivered 0 results.\n" \
               "That might be an ID identification issue. Would you like to retry using rdfs:label matching?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes I'd like that", key="retry_yes"):
            st.session_state.retry_with_label = True
            st.session_state.waiting_for_retry_decision = False
            st.session_state.retry_decision_made = True

    with col2:
        if st.button("No keep the results", key="retry_no"):
            st.session_state.waiting_for_retry_decision = False
            st.session_state.retry_decision_made = True
            st.session_state.dialog_history.append({
                "role": "assistant",
                "content": "Okay keeping the results - no retry performed.",
                "sparql_code": st.session_state.last_sparql_code,
                "table": pd.DataFrame().to_dict(),
                "time": 0
            })
            st.session_state.last_sparql_code = None
            st.session_state.last_answer = None

# Check if user has accepted the retry with label
elif st.session_state.retry_decision_made and st.session_state.retry_with_label and st.session_state.last_question:
    user_quest = st.session_state.last_question
    backend_used = "RDFS_FALLBACK"
    st.session_state.retry_with_label = False
    st.session_state.retry_decision_made = False

# Always show chat input when not waiting for retry decision
if not st.session_state.waiting_for_retry_decision and not user_quest:
    user_input = st.chat_input("How can I assist you?")
    if user_input:
        user_quest = user_input

# Process the question if we have one
if user_quest:
    start_time = time.time()
    st.session_state.last_question = user_quest

    with st.chat_message("user"):
        st.markdown(user_quest)

    # Get the answer based on backend choice
    if backend_used == "RDFS_FALLBACK":
        st.write(f"Retrying with rdfs:label for: {user_quest}")
        answer = lm.get_llm_response_rdfs(user_quest,st.session_state.dialog_history)
    else:    
        if backend_choice == "RAG Model":   
            answer = lm.get_llm_response(user_quest, st.session_state.dialog_history)
            if answer == "__NO_ENTITY_FOUND__":
                st.session_state.waiting_for_retry_decision = True
                st.rerun()
        else:
            search_terms_with_roles = sa.convert_query_to_wikidata_search(user_quest)
            search_terms_with_roles = sa.normalize_roles(search_terms_with_roles)
            term_strings = [entry["term"] for entry in search_terms_with_roles]
            search_results_dict = sa.query_search_api(term_strings)
            wikidata_ids = sa.extract_ids_per_term(search_results_dict)
            wikidata_entities = sa.get_wikidata_descriptions(wikidata_ids)
            answer = sa.natural_language_to_sparql(user_quest, wikidata_entities, search_terms_with_roles)
            print(answer)

    # Store the answer for potential retry scenarios
    st.session_state.last_answer = answer

    # Add user message to dialog history
    if backend_used == "RDFS_FALLBACK":
        st.session_state.dialog_history.append({"role": "user", "content": user_quest + " (rdfs retry)"})
    else:
        st.session_state.dialog_history.append({"role": "user", "content": user_quest})

    # Display the answer
    with st.chat_message("assistant"):
        sparql_code = None
        if "SPARQL:" in answer:
            reasoning_part, sparql_code = answer.split("SPARQL:", 1)
            typewriter(reasoning_part.strip(), 10)
            st.code(sparql_code.strip(), language="sparql")
        else:
            extracted_sparql = cs.extract_sparql_from_response(answer)
            if extracted_sparql:
                sparql_start = answer.find(extracted_sparql)
                if sparql_start > 0:
                    reasoning_part = answer[:sparql_start].strip()
                    typewriter(reasoning_part, 10)
                sparql_code = extracted_sparql
                st.code(sparql_code, language="sparql")
            else:
                sparql_code = answer.strip()
                st.code(sparql_code, language="sparql")

    # Store the SPARQL code for potential retry scenarios
    st.session_state.last_sparql_code = sparql_code

    # Execute the SPARQL query
    query = cs.clean_query(sparql_code)
    if not query:
        st.error("❌ No valid SPARQL query extracted from LLM response.")
        st.stop()

    headers = {"Accept": "application/sparql-results+json"}
    
    r = requests.get(url, params={'format': 'json', 'query': query}, headers=headers)
    if not r.ok:
        st.error(f"❌ SPARQL endpoint returned an error: {r.status_code}")
        st.text(r.text)
        st.stop()

    try:
        data = r.json()
    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse JSON: {str(e)}")
        st.text(r.text)
        st.stop()

    bindings = data.get("results", {}).get("bindings", [])
    flat_rows = [{k: v["value"] for k, v in row.items()} for row in bindings]
    
    df = pd.DataFrame(flat_rows)
    
    # Check if we got 0 results and should offer retry
    if len(df) == 0 and backend_used != "RDFS_FALLBACK":
        # Display results for the failed query but don't add to history yet
        with st.chat_message("assistant"):
            st.table(df)
            st.write(f"Number of results: {len(df)}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Question was answered in {round(elapsed_time, 2)} seconds")
        
        st.session_state.waiting_for_retry_decision = True
        st.rerun()  # Rerun to show the retry buttons
    else:
        # Display results for successful queries or retry results
        with st.chat_message("assistant"):
            st.table(df)
            st.write(f"Number of results: {len(df)}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Question was answered in {round(elapsed_time, 2)} seconds")

        # Add assistant message to dialog history
        st.session_state.dialog_history.append({
            "role": "assistant",
            "content": answer,
            "table": df.to_dict(),
            "time": elapsed_time
        })
        
        # Reset all retry-related session state after successful completion
        st.session_state.waiting_for_retry_decision = False
        st.session_state.retry_decision_made = False
        st.session_state.retry_with_label = False
        st.session_state.last_sparql_code = None
        st.session_state.last_answer = None

    # Auto scroll (moved outside the chat message context and with longer delay)
    st.components.v1.html("""
    <script>
        function scrollToBottom() {
            setTimeout(function() {
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
                        setTimeout(tryScroll, 300);
                    } else if (attempts < 10) {
                        attempts++;
                        setTimeout(tryScroll, 300);
                    }
                }

                tryScroll();
            }, 500);  // Initial delay to let content render
        }

        scrollToBottom();
    </script>
    """, height=0)