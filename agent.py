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

    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    full_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    prompt = f"{full_context}\nuser: {user_input}\nBerikan jawaban singkat:"

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
    relevant_docs = vectorstore.similarity_search(user_input, k=5)

    if not relevant_docs:
        return "Maaf, saya tidak menemukan informasi yang relevan untuk menjawab pertanyaan Anda."

    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    full_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    prompt = (
        f"Berikut adalah beberapa informasi yang relevan:\n\n"
        f"{context_text}\n\n"
        f"Percakapan sebelumnya:\n{full_context}\n\n"
        f"user: {user_input}\n\n"
        f"Jawaban:\n"
        f"Berikan jawaban hanya berdasarkan informasi di atas. "
        f"Jika memungkinkan, salin langsung kalimat dari informasi tersebut tanpa mengubah gaya bahasa. "
        f"Jangan menambahkan opini atau informasi di luar konteks. "
        f"Jika informasi tidak ditemukan, jawab dengan jujur 'Informasi tidak ditemukan dalam dokumen di atas.'"
    )

    try:
        result = agent.invoke({"input": prompt}, config={"handle_parsing_errors": True})
        answer = result.get("output", "").strip()
    except Exception as e:
        return f"Terjadi kesalahan saat memproses jawaban: {str(e)}"

    if not answer:
        return "Maaf, saya belum bisa memberikan jawaban yang tepat untuk pertanyaan tersebut."

    metrics = evaluate_metrics(context_text, user_input, answer)

    table = (
        f"| Pertanyaan | Precision | Recall | F1 | Relevancy | Similarity | Faithfulness | Correctness |\n"
        f"|---|---|---|---|---|---|---|---|\n"
        f"| `{user_input}` | {metrics['precision']}% | {metrics['recall']}% "
        f"| {metrics['f1']}% | {metrics['relevancy']}% | {metrics['similarity']}% "
        f"| {metrics['faithfulness']}% | {metrics['correctness']}% |"
    )

    return f"{answer}\n\n**Metrik Evaluasi**\n{table}"



