from langchain.vectorstores import FAISS
from langchain.schema import Document
from datetime import datetime
from embeddings import embedding
import json

with open('data/news_articles.json', 'r') as file:
    news_articles = json.load(file)


def save_news():
    docs = []

    for article in news_articles:
        if "date" in article:
            date_obj = datetime.strptime(article["date"], "%Y-%m-%d")
            metadata = {"title": article["title"], "date": date_obj, "type": article["type"]}
        else:
            metadata = {"title": article["title"], "type": article["type"]}

        doc = Document(
            page_content=article["content"],
            metadata=metadata
        )
        docs.append(doc)

    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local("news_index")

def load_news_store():
    return FAISS.load_local("news_index", embedding, allow_dangerous_deserialization=True)

# Jalankan pertama kali untuk menyimpan data berita
if __name__ == "__main__":
    save_news()
