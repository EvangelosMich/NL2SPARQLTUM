import time
import spacy

start = time.time()
nlp = spacy.load("en_core_web_sm")
print(f"spaCy load time: {time.time() - start:.2f} seconds")