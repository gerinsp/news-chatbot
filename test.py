from news_store import load_news_store

vectorstore = load_news_store()

docs = vectorstore.similarity_search("pencak silat Kabupaten Kudus", k=5)
for i, doc in enumerate(docs):
    print(f"\n--- DOC {i} ---\n{doc.page_content}")

