from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

embedding = HuggingFaceEmbeddings(model_name="model/sentence-transformers/all-MiniLM-L6-v2")
sentence_embedding = SentenceTransformer(model_name_or_path="model/sentence-transformers/all-MiniLM-L6-v2")

