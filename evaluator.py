from sentence_transformers import SentenceTransformer, util
import nltk
import os
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

model = SentenceTransformer("./model/sentence-transformers/all-MiniLM-L6-v2")

def evaluate_metrics(context: str, question: str, answer: str):
    context = context.strip()
    answer = answer.strip()

    punkt_param = PunktParameters()
    punkt_tokenizer = PunktSentenceTokenizer(punkt_param)
    sentences = punkt_tokenizer.tokenize(context)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    answer_embedding = model.encode(answer, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(answer_embedding, sentence_embeddings)[0]
    best_score = float(cosine_scores.max())

    similarity_score = best_score
    precision = similarity_score
    recall = similarity_score
    f1 = similarity_score
    faithfulness = similarity_score
    relevancy = similarity_score
    correctness = 1.0 if similarity_score > 0.6 else 0.0

    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "relevancy": round(relevancy * 100, 2),
        "similarity": round(similarity_score * 100, 2),
        "faithfulness": round(faithfulness * 100, 2),
        "correctness": round(correctness * 100, 2)
    }