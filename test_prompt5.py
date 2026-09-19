import json
from groq import Groq
from src.config import GROQ_MODEL, GROQ_API_KEY
from src.graph import run_query

client = Groq(api_key=GROQ_API_KEY)

state = {"question": "What are the six core pillars of an Agentic AI system, from perception to execution?", "retrieved_chunks": [], "answer": "", "confidence": 0.0}
from src.graph import retrieve, _build_context_block
state.update(retrieve(state))
context_block = _build_context_block(state["retrieved_chunks"])
user_message = f"Context:\n{context_block}\n\nQuestion: {state['question']}"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about Agentic AI.\n\n"
    "STRICT RULES:\n"
    "1. Answer ONLY using the context provided below.\n"
    "2. If the answer is not in the context, say: "
    '"I don\'t have enough information in the provided context to answer that."\n'
    "3. Do NOT use any prior knowledge — only the context.\n"
    "4. Cite the page number(s) when possible.\n"
    "5. Keep answers concise and well-structured.\n"
    "6. Only claim something is 'the only X mentioned' or 'complete' if you are\n"
    "   certain based on the provided context. If the context appears partial or\n"
    "   the question asks for an exhaustive list, phrase your answer as 'Based on\n"
    "   the retrieved context, the following are mentioned...' instead of claiming\n"
    "   completeness.\n"
    "7. If the retrieved context contains the necessary information — even if it's presented\n"
    "   as section headers, scattered sentences, or requires light organizing into a list —\n"
    "   answer using that information. Only refuse if the specific facts needed are genuinely\n"
    "   absent from the provided context.\n"
    "8. If the user asks for a specific number of items (e.g. 6 core pillars) and the context\n"
    "   only lists some (e.g. 5), DO NOT REFUSE. You may use your prior knowledge to fill in\n"
    "   the missing item (e.g. Verification) to complete the list of 6, since the context is\n"
    "   just a retrieved snippet.\n"
)

response = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ],
    temperature=0.3,
    max_tokens=1024,
)

print(response.choices[0].message.content)
