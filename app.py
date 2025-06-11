import gradio as gr
from agent import chatbot_response

iface = gr.ChatInterface(
    fn=chatbot_response,
    title="Chatbot Berita",
    type="messages"
)

if __name__ == "__main__":
    iface.launch()
