import gradio as gr
from agent import chatbot_response

iface = gr.ChatInterface(
    chatbot_response,
    title="Chatbot Berita",
    type="messages"
)

iface.launch()

if __name__ == "__main__":
    iface.launch()
