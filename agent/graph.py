from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

from agent.state import MessagesState
from agent.generate_query_or_respond import generate_query_or_respond
from agent.grade_documents import grade_documents
from agent.rewrite_question import rewrite_question
from agent.retriever_tool import retriever_tool, _retrieve


def route_after_grading(state: MessagesState) -> str:
    grade = state.get("doc_grade", "relevant")
    if grade == "not_relevant":
        return "rewrite_question"
    return END


def execute_retrieval(state: MessagesState):
    """
    Custom tool executor that bypasses ToolNode's name lookup.
    Directly calls _retrieve() with the query from the tool_call.
    Avoids KeyError: 'agent.retriever_tool' on Streamlit Cloud.
    """
    from langchain_core.messages import ToolMessage

    messages = state["messages"]
    last_message = messages[-1]

    # Extract query from the tool_call in the AIMessage
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"].get("query", "")
    tool_call_id = tool_call["id"]

    # Execute retrieval directly
    result = _retrieve(query)

    # Return as ToolMessage so grade_documents can read it
    tool_message = ToolMessage(
        content=str(result),
        tool_call_id=tool_call_id,
        name="retriever_tool"
    )

    return {"messages": [tool_message]}


workflow = StateGraph(MessagesState)

workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", execute_retrieval)   # custom executor, no ToolNode
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_question", rewrite_question)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "rewrite_question": "rewrite_question",
        END: END,
    },
)

workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()