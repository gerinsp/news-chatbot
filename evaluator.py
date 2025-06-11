import evaluate
from sentence_transformers import SentenceTransformer, util
from embeddings import sentence_embedding as embedding

# Muat evaluator HF
rouge = evaluate.load("rouge")

def evaluate_metrics(context_text: str, question: str, answer: str) -> dict:
    # 1. ROUGE score (F1 saja, karena evaluate tidak beri precision/recall terpisah)
    rouge_res = rouge.compute(predictions=[answer], references=[context_text])
    f1 = rouge_res["rouge1"]

    # Kita pakai F1 sebagai proxy untuk precision dan recall
    precision = f1
    recall = f1

    # 2. Similarity (cosine) antara context dan jawaban
    emb_ctx = embedding.encode(context_text, convert_to_tensor=True)
    emb_ans = embedding.encode(answer, convert_to_tensor=True)
    cosine = util.pytorch_cos_sim(emb_ctx, emb_ans).item()

    # 3. Relevansi = cosine similarity
    relevancy = cosine

    # 4. Faithfulness & Correctness
    faithfulness = recall
    correctness = f1

    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "relevancy": round(relevancy * 100, 2),
        "similarity": round(cosine * 100, 2),
        "faithfulness": round(faithfulness * 100, 2),
        "correctness": round(correctness * 100, 2),
    }
