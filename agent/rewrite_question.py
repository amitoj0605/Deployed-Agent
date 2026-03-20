# agent/rewrite_question.py
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from utils.logger import log
from dotenv import load_dotenv

load_dotenv()

response_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=128,
    api_key=os.getenv("GROQ_API_KEY")
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state):
    messages = state["messages"]
    rewrite_count = state.get("rewrite_count", 0)
    question = messages[0].content
    log(f"Rewriting question (attempt {rewrite_count + 1}): '{question}'")
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    log(f"Rewritten query: '{response.content}'")
    return {
        "messages": [HumanMessage(content=response.content)],
        "rewrite_count": rewrite_count + 1
    }