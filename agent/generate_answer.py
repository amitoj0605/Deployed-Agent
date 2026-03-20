# agent/generate_answer.py
import os
import time
from utils.logger import log
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Fallback model — not used in graph, only GENERATE_PROMPT is imported by chat_app.py
response_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=300,
    api_key=os.getenv("GROQ_API_KEY")
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If the answer is not in the context, say you don't know. "
    "If the question is a follow-up, use conversation history to understand context. "
    "Keep the answer concise (max three sentences).\n\n"
    "Question: {question}\n\n"
    "Conversation history:\n{history}\n\n"
    "Context:\n{context}"
)


def generate_answer(state):
    log("Node: generate_answer started")
    start = time.time()
    messages = state["messages"]
    question = messages[0].content
    context = messages[-1].content[:2000]
    log(f"Context length: {len(context)} characters")
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke(prompt)
    log(f"LLM response time: {time.time() - start:.2f}s")
    log("Final answer generated")
    return {"messages": [response]}