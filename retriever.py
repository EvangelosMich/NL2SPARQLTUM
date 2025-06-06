import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class example_retriever:
    def __init__(self,path="examples/examples.json"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        with open(path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)
        self.questions = [ex["question"] for ex in self.examples]
        self.embeddings = self.model.encode(self.questions, normalize_embeddings=True)

    def retrieve(self, new_question, top_k=3):
        query_emb = self.model.encode([new_question], normalize_embeddings=True)
        scores = cosine_similarity(query_emb, self.embeddings)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.examples[i] for i in top_indices]