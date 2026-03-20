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


# Phrases that should NEVER trigger retrieval
CONVERSATIONAL_PHRASES = [
    "hi", "hello", "hey", "how are you", "what's up", "whats up",
    "good morning", "good evening", "good night", "thanks", "thank you",
    "bye", "goodbye", "who are you", "what can you do", "help",
    "ok", "okay", "cool", "nice", "great", "awesome"
]


def is_conversational(text: str) -> bool:
    """Returns True if the message is a greeting or small talk."""
    lowered = text.lower().strip().rstrip("!?.")
    return lowered in CONVERSATIONAL_PHRASES


def generate_query_or_respond(state: MessagesState):
    """
    Decide whether to answer directly or call the retriever tool.
    Receives full conversation history via state["messages"] for memory.
    """
    messages = state["messages"]

    # Get the latest user message
    last_message = messages[-1].content if messages else ""

    # Fast path — skip LLM entirely for conversational messages
    # Saves a full Groq API call (~1-2s) for simple greetings
    if is_conversational(last_message):
        ui_response = response_model.invoke(messages)
        return {"messages": [ui_response]}

    system_prompt = SystemMessage(
        content=(
            "You are an AI assistant with a retriever_tool to search a knowledge base.\n"
            "You have access to the conversation history — use it to understand follow-up questions.\n"
            "ALWAYS call retriever_tool for ANY question about AI, agents, RAG, "
            "agentic AI, generative AI, or any factual topic.\n"
            "Only respond directly for greetings or purely conversational messages."
        )
    )

    messages_with_system = [system_prompt] + messages
    model_with_tools = response_model.bind_tools([retriever_tool])
    response = model_with_tools.invoke(messages_with_system)
    return {"messages": [response]}