from sentence_transformers import SentenceTransformer

# Download dan simpan model ke lokal
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
model.save("./model/paraphrase-multilingual-MiniLM-L12-v2")
