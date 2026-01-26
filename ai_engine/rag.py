from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# --- CORRECT IMPORTS ---
from ai_engine.retriever import generate_query_or_respond
from ai_engine.retriever import generate_answer
from ai_engine.retriever import grade_documents
from ai_engine.retriever import rewrite_question
from ai_engine.retriever import retrieve_data, retriever_tool

memory = MemorySaver()

workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

# Define edges
workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    }
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
workflow = workflow.compile(checkpointer=memory)
