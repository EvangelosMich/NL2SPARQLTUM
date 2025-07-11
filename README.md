# 🧠 NL2SPARQL: Natural Language to SPARQL Translator for Wikidata

This project provides a full-stack application that translates natural language questions into SPARQL queries using Large Language Models (LLMs), enabling non-technical users to query Wikidata interactively.

Developed by **Evangelos Michalopoulos** and **Henrique Leonel** as part of the TUM Bachelor Practical Course in Data Engineering (SoSe 2025).

---

## 🔍 Key Features

- 🔁 **Two SPARQL Generation Backends**:
  - 🧭 **Search Tool (online)**: Extracts keywords, resolves QIDs using Serper + Wikidata APIs, and generates SPARQL via GPT-4o.
  - 🧠 **RAG Model (offline)**: Uses FAISS to retrieve relevant entities and properties, adds few-shot Chain-of-Thought examples, and generates queries via Gemini Pro 2.5.

- 🧠 **LLM Reasoning Strategies**:
  - Chain-of-Thought and Self-Ask prompting styles.
  - Structured responses with both reasoning and final SPARQL output.

- 💬 **Interactive Frontend**:
  - Built with **Streamlit**.
  - Displays conversation history, answers, results, and response time.
  - Retry button enables `rdfs:label` fallback for unmatched entities.

- 🛡️ **Robust Fallback Mechanism**:
  - Automatically falls back to `rdfs:label`-based querying when entity QIDs cannot be reliably resolved.
  - Reduces hallucination of non-existent or incorrect entity IDs.

---

## 🧱 Project Structure

```bash
.
├── FrontEnd.py                # Streamlit app entry point
├── captureSparql.py           # Cleans and executes SPARQL queries
├── searchTool/
│   └── searchtool.py          # Online search + query generator using GPT and Serper
├── RAGModel/
│   ├── llmbasedbackend.py     # Offline RAG pipeline using Gemini + FAISS
│   ├── prompt_template.py     # Prompt templates for structured query generation
│   └── jsonfiles/             # Local FAISS index, entity & property DBs
├── examples/                  # Few-shot SPARQL examples for prompting
│   ├── examples.json
│   └── examples_rdfs.json
├── .env                       # API key config for OpenAI, Gemini, Serper
├── requirements.txt           # Python dependencies
├── setup.sh                   # Virtual environment + dependency installer
└── README.md




🛠️ Setup Instructions
1 Clone the repo:


git clone https://github.com/EvangelosMich/NL2SPARQLTUM.git
cd NL2SPARQLTUM

2. Set up the environment
chmod +x setup.sh
./setup.sh





3Create a .env file with your API keys:

OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
SERPER_API_KEY=your_serper_key


4 # Run from Git Bash or WSL or Linux of course

Run the Streamlit app:

source .venv/bin/activate #this step can probably be skipped but if the next command doesnt work run it from the venv
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