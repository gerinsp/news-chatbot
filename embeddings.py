from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GOOGLE_API_KEY

embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=GOOGLE_API_KEY,
)