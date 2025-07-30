from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    # Retrieval‑based
    context_precision,       # rata‑rata Precision atas dokumen yang di‐retrieve
    context_recall,          # Recall atas entitas/informasi yang dibutuhkan

    # Answer‑based
    answer_similarity,       # Semantic similarity antara jawaban & ground truth
    answer_correctness,      # Kombinasi factuality + similarity terhadap ground truth
    faithfulness,
)


from config import GOOGLE_API_KEY
from embeddings import embedding
from langchain_google_genai import GoogleGenerativeAI

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    api_key=GOOGLE_API_KEY,
)

METRICS = [
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness,
    faithfulness
]

def evaluate_metrics(query: str, contexts: list[str], answer: str, reference: str = None):
    if reference is None:
        reference = answer

    data = {
        "user_input": [query],
        "response": [answer],
        "retrieved_contexts": [contexts],
        "ground_truth": [reference],
    }
    ds = Dataset.from_dict(data)
    result = evaluate(
        llm=llm,
        embeddings=embedding,
        dataset=ds,
        metrics=METRICS,
    )

    return result.to_pandas()
