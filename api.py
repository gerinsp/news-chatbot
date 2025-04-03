from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from news_store import save_news, load_news_store
from embeddings import embedding
from langchain.vectorstores import FAISS
from datetime import datetime

app = FastAPI()

vectorstore = load_news_store()

class NewsArticle(BaseModel):
    title: str
    content: str
    date: str

def add_news_to_faiss(article: NewsArticle):
    date_obj = datetime.strptime(article.date, "%Y-%m-%d")
    doc = {
        "page_content": article.content,
        "metadata": {
            "title": article.title,
            "date": date_obj
        }
    }
    vectorstore.add_documents([doc])

@app.post("/add_news/")
async def add_news(article: NewsArticle):
    try:
        add_news_to_faiss(article)
        return {"message": "Berita berhasil ditambahkan ke FAISS"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
