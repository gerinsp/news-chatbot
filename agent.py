from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from config import GOOGLE_API_KEY
from news_store import load_news_store
from datetime import datetime
from langchain.schema import AIMessage, HumanMessage

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
    tools, chat_model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


def chatbot_response(user_input, history):
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history] + [f"user: {user_input}"])

    response = agent.invoke({"input": context})

    return response["output"]



