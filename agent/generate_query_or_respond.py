# agent/generate_query_or_respond.py
import os
from langgraph.graph import MessagesState
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from agent.retriever_tool import retriever_tool
from dotenv import load_dotenv

load_dotenv()

# Groq's llama-3.1-8b-instant is extremely fast at tool calling
# num_predict equivalent is handled by max_tokens in Groq
response_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=150,
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_query_or_respond(state: MessagesState):
    """
    Decide whether to answer directly or call the retriever tool.
    """
    system_prompt = SystemMessage(
        content=(
            "You are an AI assistant with a retriever_tool to search a knowledge base.\n"
            "ALWAYS call retriever_tool for ANY question about AI, agents, RAG, "
            "agentic AI, generative AI, or any factual topic.\n"
            "Only respond directly for greetings or purely conversational messages."
        )
    )

    messages = [system_prompt] + state["messages"]
    model_with_tools = response_model.bind_tools([retriever_tool])
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}