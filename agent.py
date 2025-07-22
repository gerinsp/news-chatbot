from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from config import GOOGLE_API_KEY
from news_store import load_news_store
from datetime import datetime
from langchain.schema import AIMessage, HumanMessage
from evaluator import evaluate_metrics

vectorstore = load_news_store()

chat_model = GoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=GOOGLE_API_KEY
    )

def get_relevant_news(query):
    filter_query = {"type": "news"}
    docs = vectorstore.similarity_search(query, k=5, filter=filter_query)
    formatted_docs = []

    for doc in docs:
        title = doc.metadata['title']
        content = doc.page_content
        date = doc.metadata['date'].strftime('%d %B %Y')  # Format tanggal: 2 April 2025
        formatted_doc = f"**{title}**\ndiposting tanggal {date}\n{content}"
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs)

def get_relevant_form(query):
    filter_query = {"type": "form"}
    docs = vectorstore.similarity_search(query, k=5, filter=filter_query)
    formatted_docs = []

    for doc in docs:
        title = doc.metadata['title']
        content = doc.page_content
        formatted_doc = f"**{title}**\n{content}"
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs)

def get_today_date(_=None):
    return datetime.today().strftime('%Y-%m-%d')

tools = [
    Tool(
        name="Berita Relevan",
        func=get_relevant_news,
        description="Gunakan ini untuk mencari berita yang terkait dengan pertanyaan."
    ),
    Tool(
        name="Form Pendaftaran Relevan",
        func=get_relevant_form,
        description="Gunakan ini untuk mencari formulir pendaftaran yang terkait dengan pertanyaan."
    ),
    Tool(
        name="Tanggal Hari ini",
        func=get_today_date,
        description="Gunakan ini untuk mengetahui tanggal hari ini dalam format YYYY-MM-DD."
    )
]

agent = initialize_agent(
    tools, chat_model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
)


def chatbot_response_api(user_input: str, history: list[dict]) -> dict:
    relevant_docs = vectorstore.similarity_search(user_input, k=5, filter={"type": "news"})

    if not relevant_docs:
        return {
            "answer": "Maaf, saya tidak menemukan informasi yang relevan untuk menjawab pertanyaan Anda."
        }

    context_text = "\n\n".join([
        f"""{doc.page_content}<br>
        Sumber: <a href="http://127.0.0.1:8000/posts/{doc.metadata['slug']}" target="_blank">
          http://127.0.0.1:8000/posts/{doc.metadata['slug']}
        </a><br>"""
        for doc in relevant_docs
        if 'slug' in doc.metadata
    ])

    full_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    prompt = (
        f"Berikut adalah beberapa informasi berita yang relevan:\n\n"
        f"{context_text}\n\n"
        f"Percakapan sebelumnya:\n{full_context}\n\n"
        f"user: {user_input}\n\n"
        f"Jawaban:\n"
        f"Berikan jawaban singkat berdasarkan informasi di atas. "
        f"Jika pengguna tampaknya ingin membaca sumber lengkapnya, dan informasi tersedia, "
        f"tampilkan link artikel yang ada di dalam teks (berupa 'Sumber: https://...'). "
        f"Jika informasi tidak ditemukan, jawab dengan jujur."
        f"Jangan hilangkan tag HTML <a></a> untuk text berupa link."
    )

    try:
        result = agent.invoke({"input": prompt})
        answer = result.get("output", "").strip()
    except Exception as e:
        return {
            "answer": f"Terjadi kesalahan saat memproses jawaban: {str(e)}"
        }

    if not answer:
        return {
            "answer": "Maaf, saya belum bisa memberikan jawaban yang tepat untuk pertanyaan tersebut."
        }

    return {
        "answer": answer
    }

def chatbot_response(user_input, history):
    # 1) Ambil dokumen relevan
    relevant_docs = vectorstore.similarity_search(user_input, k=5)
    if not relevant_docs:
        return "Maaf, saya tidak menemukan informasi yang relevan untuk menjawab pertanyaan Anda."

    # 2) Siapkan context list
    contexts = []
    for doc in relevant_docs:
        if 'slug' in doc.metadata:
            teks = doc.page_content
            link = f"http://127.0.0.1:8000/posts/{doc.metadata['slug']}"
            contexts.append(f"{teks}\n(Sumber: {link})")

    # 3) Bangun prompt untuk agent
    prior_conv = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    prompt = (
        "Berikut adalah beberapa informasi yang relevan:\n\n"
        f"{chr(10).join(contexts)}\n\n"
        f"Percakapan sebelumnya:\n{prior_conv}\n\n"
        f"user: {user_input}\n\n"
        "Jawaban:\n"
        "Jawab pertanyaan pengguna hanya berdasarkan informasi di atas. "
        "Berikan jawaban yang langsung menjawab inti pertanyaan. "
        "Jika pertanyaan memerlukan nama orang, tempat, atau organisasi, berikan hanya entitas tersebut tanpa penjelasan tambahan. "
        "Jika memungkinkan, kutip secara langsung dari informasi di atas tanpa mengubah gaya bahasa. "
        "Jangan menambahkan opini, interpretasi, atau informasi di luar konteks. "
        "Jika informasi tidak ditemukan, jawab: 'Informasi tidak ditemukan dalam dokumen di atas.'"
    )

    # 4) Panggil Agent (LangChain) yang dikonfigurasikan memakai Gemini
    try:
        result = agent.invoke({"input": prompt})
        answer = result.get("output", "").strip()
    except Exception as e:
        return f"Terjadi kesalahan saat memproses jawaban: {e}"

    if not answer:
        return "Maaf, saya belum bisa memberikan jawaban yang tepat untuk pertanyaan tersebut."

    # 5) Evaluasi dengan RAGAS
    df_metrics = evaluate_metrics(user_input, contexts, answer)
    # Ambil baris pertama
    row = df_metrics.iloc[0]

    # 6) Format tabel Markdown
    table = (
        "| Pertanyaan | Precision | Recall | Entity Recall | Answer Relevancy | Similarity | Correctness |\n"
        "|---|---|---|---|---|---|---|\n"
        f"| `{row['user_input']}` "
        f"| {row['context_precision'] * 100:.1f}% "
        f"| {row['context_recall'] * 100:.1f}% "
        f"| {row['context_entity_recall'] * 100:.1f}% "
        f"| {row['answer_relevancy'] * 100:.1f}% "
        f"| {row['semantic_similarity'] * 100:.1f}% "
        f"| {row['answer_correctness'] * 100:.1f}% |"
    )

    return f"{answer}\n\n**Metrik Evaluasi**\n{table}"



