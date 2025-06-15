import spacy

NLP = spacy.load("en_core_web_sm")
prompt = NLP("Show me the last 10 Queens of England")

prompt2 = "Show me the last 10 queens of England"



keywords = set(ent.text for ent in prompt.ents)
keywords.update(chunk.text for chunk in prompt.noun_chunks if len(chunk.text.split()) > 1)

print(keywords)