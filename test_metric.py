from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
from config import GOOGLE_API_KEY
from langchain_google_genai import GoogleGenerativeAI

evaluator_llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    api_key=GOOGLE_API_KEY,
)

context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="information not found in the above document.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
)


await context_precision.single_turn_ascore(sample)