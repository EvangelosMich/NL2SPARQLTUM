# 🧠 NL2SPARQL: Natural Language to SPARQL Translator for Wikidata

This project provides a full-stack application that translates natural language questions into SPARQL queries using Large Language Models (LLMs), enabling non-technical users to query Wikidata interactively.

Developed by **Evangelos Michalopoulos** and **Henrique Leonel** as part of the TUM Bachelor Practical Course in Data Engineering (SoSe 2025).

---

## 🔍 Key Features

- 🔁 **Two SPARQL Generation Backends**:
  - **Search Tool** (online): Extracts keywords, resolves QIDs using Serper + Wikidata APIs, and generates SPARQL via GPT-4o.
  - **RAG Model** (offline): Retrieves entities and properties using FAISS, adds few-shot Chain-of-Thought examples, and generates queries via Gemini Pro 2.5.

- 🧠 **LLM Reasoning Strategies**:
  - Chain-of-Thought & Self-Ask prompting styles.
  - SPARQL-only output with reasoning explanation included.

- 💬 **Interactive Frontend**:
  - Built with Streamlit.
  - Displays conversation history, answers, tables, and response time.
  - Retry button for using `rdfs:label` fallback in case of errors or hallucinations.

- 🛡️ **Fallback Mechanism**:
  - `rdfs:label`-based querying if QID lookup fails.
  - Prevents hallucination of invalid IDs for unseen or rare entities.

---

## 🧱 Project Structure

```bash
.
├── FrontEnd.py                # Streamlit app entry point
├── captureSparql.py           # Cleans and runs SPARQL queries
├── searchTool/
│   └── searchtool.py          # Online search + query generator using GPT and Serper
├── RAGModel/
│   ├── llmbasedbackend.py     # Offline RAG implementation using Gemini and FAISS
│   ├── prompt_template.py     # Prompt structure for Gemini
│   └── jsonfiles/             # Offline FAISS index, entity/property databases
├── examples/                  # Few-shot SPARQL examples for Chain-of-Thought prompting
│   ├── examples.json
│   └── examples_rdfs.json
├── .env                       # API keys for OpenAI, Gemini, Serper
└── README.md



🛠️ Setup Instructions
1 Clone the repo:


git clone https://github.com/EvangelosMich/NL2SPARQLTUM.git
cd NL2SPARQLTUM

2
Install dependencies:


pip install -r requirements.txt



3Create a .env file with your API keys:

OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
SERPER_API_KEY=your_serper_key


4
Run the Streamlit app:


streamlit run FrontEnd.py


Evaluation Overview
Evaluation was conducted using handcrafted benchmark queries across:

1-hop, 2-hop, and 3-hop questions

Domains like people, events, movies, etc.

Metrics: Precision, Recall, and F1 Score

Fallback via rdfs:label proved crucial in cases with entities not covered by the offline index.



Contact
Evangelos Michalopoulos – michalopoulos.evangelos03@gmail.com

Henrique Leonel – hleonel2013@gmail.com