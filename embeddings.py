from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

embedding = HuggingFaceEmbeddings(model_name="model/paraphrase-multilingual-MiniLM-L12-v2")
sentence_embedding = SentenceTransformer(model_name_or_path="model/paraphrase-multilingual-MiniLM-L12-v2")

