from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from news_store import save_news, load_news_store
from datetime import datetime
from agent import chatbot_response_api
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            "date": date_obj,
            "type": "news"
        }
    }
    vectorstore.add_documents([doc])

@app.get("/")
def read_root():
    return {"message": "API siap digunakan"}

class ChatRequest(BaseModel):
    user_input: str
    history: list[dict]  # format: [{"role": "user", "content": "..."}, ...]

@app.post("/chat/")
async def chat_endpoint(req: ChatRequest):
    try:
        response = chatbot_response_api(req.user_input, req.history)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_news/")
async def add_news(article: NewsArticle):
    try:
        add_news_to_faiss(article)
        return {"message": "Berita berhasil ditambahkan ke FAISS"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
