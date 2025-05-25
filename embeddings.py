from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="model/sentence-transformers/all-MiniLM-L6-v2")

