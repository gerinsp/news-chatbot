from langchain.vectorstores import FAISS
from langchain.schema import Document
from datetime import datetime
from embeddings import embedding
from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://root@localhost:3306/portal_berita")

def fetch_news_from_db():
    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT 
                    post_title as title, 
                    post_content as content, 
                    published_at as date 
                FROM posts
                WHERE post_status = 'Published'
                """))
        articles = []
        for row in result:
            article = {
                "title": row.title,
                "content": row.content,
                "type": 'news',
                "date": row.date.strftime("%Y-%m-%d") if row.date else None
            }
            articles.append(article)
        return articles

def save_news():
    docs = []
    news_articles = fetch_news_from_db()

    for article in news_articles:
        if article["date"]:
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
