from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("model/sentence-transformers/all-MiniLM-L6-v2")
