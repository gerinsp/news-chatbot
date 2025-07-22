from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    # Retrieval‑based
    context_precision,       # rata‑rata Precision atas dokumen yang di‐retrieve
    context_recall,          # Recall atas entitas/informasi yang dibutuhkan
    context_entity_recall,   # Recall khusus pada entitas yang tercantum

    # Answer‑based
    answer_relevancy,        # Seberapa relevan jawaban terhadap pertanyaan
    answer_similarity,       # Semantic similarity antara jawaban & ground truth
    answer_correctness,      # Kombinasi factuality + similarity terhadap ground truth
)


from config import GOOGLE_API_KEY
from embeddings import embedding
from langchain_google_genai import GoogleGenerativeAI

# 1) Inisialisasi LLM Gemini (sesuaikan model ID & auth kamu)
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    api_key=GOOGLE_API_KEY,
)

METRICS = [
    context_precision,
    context_recall,
    context_entity_recall,
    answer_relevancy,
    answer_similarity,
    answer_correctness
]

def evaluate_metrics(query: str, contexts: list[str], answer: str, reference: str = None):
    """
    Panggil RAGAS evaluate() dan kembalikan Pandas DataFrame.
    """
    if reference is None:
        reference = answer

    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [reference],  # ganti jika punya ground truth berbeda
    }
    ds = Dataset.from_dict(data)
    result = evaluate(
        llm=llm,
        embeddings=embedding,
        dataset=ds,
        metrics=METRICS,
    )
    # Konversi ke Pandas untuk kemudahan indexing
    return result.to_pandas()
