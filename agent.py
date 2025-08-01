from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from config import GOOGLE_API_KEY
from news_store import load_news_store
from datetime import datetime
from evaluator import evaluate_metrics
from bs4 import BeautifulSoup

vectorstore = load_news_store()

chat_model = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GOOGLE_API_KEY,
        temperature=0.1
    )

def get_relevant_news(query):
    filter_query = {"type": "news"}
    docs = vectorstore.similarity_search(query, k=5, filter=filter_query)
    formatted_docs = []

    for doc in docs:
        title = doc.metadata['title']
        content = doc.page_content
        date = doc.metadata['date'].strftime('%d %B %Y')
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
    relevant_docs = vectorstore.similarity_search(user_input, k=5)
    for i, doc in enumerate(relevant_docs):
        if "pencak silat Kabupaten Kudus" in doc.page_content:
            print(f"\n--- DOC {i} ---\n{doc.page_content}")

    if not relevant_docs:
        return {
            "answer": "Maaf, saya tidak menemukan informasi yang relevan untuk menjawab pertanyaan Anda."
        }

    clean_context_list = []

    linked_context_list = []

    for doc in relevant_docs:
        clean_text = BeautifulSoup(doc.page_content, "html.parser").get_text(separator=" ")
        clean_context_list.append(clean_text)

        if 'slug' in doc.metadata:
            linked_context_list.append(
                f"""{clean_text}<br>
                Sumber: <a href="http://127.0.0.1:8000/posts/{doc.metadata['slug']}" target="_blank">
                  http://127.0.0.1:8000/posts/{doc.metadata['slug']}
                </a><br>"""
            )

    context_text_for_prompt = "\n\n".join(linked_context_list)

    full_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    prompt = (
        f"Berikut adalah beberapa informasi berita yang relevan:\n\n"
        f"{context_text_for_prompt}\n\n"
        f"Percakapan sebelumnya:\n{full_context}\n\n"
        f"user: {user_input}\n\n"
        "Jawaban:\n"
        "Berikan jawaban singkat berdasarkan informasi di atas. "
        "Jika pengguna tampaknya ingin membaca sumber lengkapnya, dan informasi tersedia, "
        "tampilkan link artikel yang ada di dalam teks (berupa 'Sumber: https://...'). "
        "Jika informasi tidak ditemukan, jawab dengan jujur."
        "Jangan hilangkan tag HTML <a></a> untuk text berupa link."
        "Jika informasi tidak ditemukan, jawab: 'Final Answer: Informasi tidak ditemukan dalam dokumen di atas.'\n"
        "Tulis jawaban diawali dengan: Final Answer:"
    )

    try:
        result = agent.invoke({"input": prompt}, handle_parsing_errors=True)
        answer = result.get("output", "").strip()
    except Exception as e:
        return {
            "answer": f"Terjadi kesalahan saat memproses jawaban: {str(e)}"
        }

    return {
        "answer": answer
    }

def chatbot_response(user_input, history):
    relevant_docs = vectorstore.similarity_search(user_input, k=5)
    if not relevant_docs:
        return "Maaf, saya tidak menemukan informasi yang relevan untuk menjawab pertanyaan Anda."

    contexts = []
    for doc in relevant_docs:
        teks = doc.page_content
        clean_text = BeautifulSoup(doc.page_content, "html.parser").get_text(separator=" ")
        contexts.append(f"{clean_text}")

    prior_conv = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    prompt = f"""
            Berikut adalah beberapa informasi yang relevan:

            {chr(10).join(contexts)}

            Percakapan sebelumnya:
            {prior_conv}

            user: {user_input}

            Jawaban:
            - Pilih tepat satu **kalimat** dari teks di atas yang menjadi jawaban.
            - Jawaban harus **persis sama** dengan kalimat di konteks (tanpa berubah).
            - Tidak perlu menambahkan kata lain: outputkan **hanya** kalimat jawaban.
            - Kalau tidak ditemukan, tulis: “Final Answer: Informasi tidak ditemukan dalam dokumen di atas.”

            Tulis jawaban diawali dengan: Final Answer:
            """

    try:
        result = agent.invoke({"input": prompt}, handle_parsing_errors=True)
        answer = result.get("output", "").strip()
    except Exception as e:
        return f"Terjadi kesalahan saat memproses jawaban: {e}"

    if not answer:
        return "Maaf, saya belum bisa memberikan jawaban yang tepat untuk pertanyaan tersebut."

    print(f"conteks: {contexts}")
    df_metrics = evaluate_metrics(user_input, contexts, answer)
    row = df_metrics.iloc[0]

    table = (
        "| Pertanyaan | Precision | Recall | Similarity | Correctness | Faithfulness |\n"
        "|---|---|---|---|---|---|\n"
        f"| `{row['user_input']}` "
        f"| {row['context_precision'] * 100:.1f}% "
        f"| {row['context_recall'] * 100:.1f}% "
        f"| {row['semantic_similarity'] * 100:.1f}% "
        f"| {row['answer_correctness'] * 100:.1f}% "
        f"| {row['faithfulness'] * 100:.1f}% |"
    )

    return f"{answer}\n\n**Metrik Evaluasi**\n{table}"



