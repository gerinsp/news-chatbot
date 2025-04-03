from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from config import HUGGINGFACEHUB_API_TOKEN

embedding = HuggingFaceInferenceAPIEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    api_key=HUGGINGFACEHUB_API_TOKEN,
)
