from langchain.tools import tool
from dotenv import load_dotenv
import os
from langgraph.graph import MessagesState
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS

# --- SAFE IMPORT: Handle missing library gracefully ---
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("⚠️ WARNING: google-generativeai library not found.")

# --- LAZY LOADER IMPORT ---
from ai_engine.embeddings import get_embedding_model

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- SAFE CONFIGURATION ---
if HAS_GEMINI and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"⚠️ Error configuring Gemini: {e}")

# --- HELPER: Get Model Safely ---
def get_gemini_model():
    if not HAS_GEMINI:
        raise ImportError("Google Generative AI library is missing.")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing in Environment Variables.")
    return genai.GenerativeModel("gemini-2.5-flash")

# --- TOOL DEFINITION ---
@tool
def retrieve_data(query: str) -> str:
    """Retrieve data from the saved vectorstore on disk."""
    print(f"🔍 RETRIEVE_DATA: {query}")

    try:
        from django.conf import settings
        faiss_path = os.path.join(settings.BASE_DIR, "faiss_index")

        if not os.path.exists(faiss_path):
            return "❌ No document found. Please upload a PDF first."

        # Lazy load the embedding model
        embedding_model = get_embedding_model()

        vector_store = FAISS.load_local(
            faiss_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        Retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        docs = Retriever.invoke(query)

        if not docs:
            return "I couldn't find relevant information."

        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

        # Generate answer
        return get_gemini_model().generate_content(prompt).text

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return f"I encountered an error: {str(e)}"

retriever_tool = retrieve_data

# --- CHAT LOGIC ---
def generate_query_or_respond(state: MessagesState):
    try:
        last_message = state["messages"][-1]
        user_content = last_message.content.lower() if hasattr(last_message, 'content') else str(last_message).lower()

        keywords = ["what", "how", "explain", "summary", "tell me", "?"]

        if any(k in user_content for k in keywords):
            answer = retrieve_data.invoke({"query": user_content})
            response = AIMessage(content=answer)
        else:
            text = get_gemini_model().generate_content(user_content).text
            response = AIMessage(content=text)

        return {"messages": [response]}
    except Exception as e:
        # Fallback if AI fails so server doesn't crash
        return {"messages": [AIMessage(content=f"⚠️ AI Error: {str(e)}")]}

# --- PLACEHOLDER FUNCTIONS (To prevent import errors) ---
def grade_documents(state): return "generate_answer"
def rewrite_question(state): return {"messages": [HumanMessage(content=state["messages"][0].content)]}
def generate_answer(state): return generate_query_or_respond(state)
