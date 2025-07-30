from langchain.vectorstores import FAISS
from langchain.schema import Document
from datetime import datetime
from embeddings import embedding
from sqlalchemy import create_engine, text
import json
from bs4 import BeautifulSoup

with open('data/news_articles.json', 'r', encoding='utf-8') as file:
    news_form = json.load(file)

engine = create_engine("mysql+pymysql://root@localhost:3306/portal_berita")

def fetch_news_from_db():
    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT 
                    post_title as title, 
                    post_content as content, 
                    slug,
                    published_at as date 
                FROM posts
                WHERE post_status = 'Published'
                """))
        articles = []
        for row in result:
            article = {
                "title": row.title,
                "content": row.content,
                "slug": row.slug,
                "type": 'news',
                "date": row.date.strftime("%Y-%m-%d") if row.date else None
            }
            articles.append(article)
        return articles


def save_news():
    news_articles = fetch_news_from_db()
    all_articles = news_articles + news_form

    docs = []

    for article in all_articles:
        if not article.get("content"):
            continue

        clean_content = BeautifulSoup(article["content"], "html.parser").get_text(separator=" ")

        article["content"] = clean_content

        try:
            doc = create_document(article)
            docs.append(doc)
        except Exception as e:
            print(f"Gagal membuat dokumen untuk: {article.get('title')} - {e}")

    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local("news_index")

    print(f"✅ Total dokumen disimpan di FAISS: {len(vectorstore.docstore._dict)}")

    for doc in vectorstore.docstore._dict.values():
        if "fspmi" in doc.page_content.lower():
            print("✅ DOKUMEN FSPMI DITEMUKAN")
            print("Title:", doc.metadata.get("title"))
            print("Slug:", doc.metadata.get("slug"))
            print("Excerpt:", doc.page_content[:300])
            print("=" * 50)

def create_document(article):
    metadata = {
        "title": article["title"],
        "type": article["type"],
        "slug": article.get("slug")
    }
    if article.get("date"):
        try:
            date_obj = datetime.strptime(article["date"], "%Y-%m-%d")
            metadata["date"] = date_obj
        except ValueError:
            pass
    return Document(page_content=article["content"], metadata=metadata)

def load_news_store():
    return FAISS.load_local("news_index", embedding, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    save_news()
