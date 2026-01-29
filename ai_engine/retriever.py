from langchain.tools import tool
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# --- 1. CORRECT IMPORT: Import the FUNCTION, not the variable ---
from ai_engine.embeddings import get_embedding_model

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- 2. SETUP LLM (Using Native Gemini) ---
# We use a function to get the model to be safe, though Gemini client is lightweight
def get_gemini_model():
    return genai.GenerativeModel("gemini-2.5-flash")

# --- 3. DEFINE THE RETRIEVER TOOL ---
@tool
def retrieve_data(query: str) -> str:
    """Retrieve data from the saved vectorstore on disk."""
    print(f"\n{'='*60}")
    print(f"🔍 RETRIEVE_DATA CALLED")
    print(f"📝 Query: {query}")
    print(f"{'='*60}\n")

    try:
        from django.conf import settings
        faiss_path = os.path.join(settings.BASE_DIR, "faiss_index")

        if not os.path.exists(faiss_path):
            return "❌ No document found. Please upload a PDF first."

        # --- FIX: Load the "Brain" using the LAZY loader ---
        embedding_model = get_embedding_model()

        vector_store = FAISS.load_local(
            faiss_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        Retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        docs = Retriever.invoke(query)

        if len(docs) == 0:
            return "I couldn't find relevant information in the document."

        # Combine context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create Prompt
        prompt = f"""You are a Study Assistant. Use the Context below to answer the Question.
        If the answer isn't in the context, say 'I don't see that in your notes'.

        Context: {context}

        Question: {query}

        Answer:"""

        # Call Gemini
        response = get_gemini_model().generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return f"I encountered an error: {str(e)}"

retriever_tool = retrieve_data

# --- 4. CHAT LOGIC ---
def generate_query_or_respond(state: MessagesState):
    """Decide whether to retrieve data or just chat."""
    last_message = state["messages"][-1]

    # Simple keyword check to save AI calls
    question_keywords = ["what", "how", "explain", "summary", "tell me", "describe", "?"]
    user_content = last_message.content.lower() if hasattr(last_message, 'content') else str(last_message).lower()

    if any(keyword in user_content for keyword in question_keywords):
        # Use the retriever
        answer = retrieve_data.invoke({"query": user_content})
        response = AIMessage(content=answer)
    else:
        # Just chat normally
        response_text = get_gemini_model().generate_content(user_content).text
        response = AIMessage(content=response_text)

    return {"messages": [response]}

# --- 5. HELPER FUNCTIONS ---
def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Simplified grader."""
    return "generate_answer"  # Keep it simple for now to prevent loops

def rewrite_question(state: MessagesState) -> dict:
    """Rewrite question."""
    question = state["messages"][0].content
    response = get_gemini_model().generate_content(f"Rewrite for search: {question}").text
    return {"messages": [HumanMessage(content=response)]}

def generate_answer(state: MessagesState):
    """Generate final answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = f"Answer this question using the context: {question}\nContext: {context}"
    response = get_gemini_model().generate_content(prompt).text

    return {"messages": [AIMessage(content=response)]}
