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

# --- IMPORT EMBEDDINGS CORRECTLY ---
from ai_engine.embeddings import embedding_model

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# --- 1. SETUP LLM (Using Native Gemini) ---
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


# --- 2. DEFINE THE RETRIEVER TOOL ---
active_vector_store = None

@tool
def retrieve_data(query: str) -> str:
    """Retrieve data from the saved vectorstore on disk."""
    print(f"\n{'='*60}")
    print(f"🔍 RETRIEVE_DATA CALLED")
    print(f"📝 Query: {query}")
    print(f"{'='*60}\n")

    try:
        # Import Django settings
        from django.conf import settings

        # Define FAISS path (same as in embeddings.py)
        faiss_path = os.path.join(settings.BASE_DIR, "faiss_index")
        print(f"📁 Looking for FAISS index at: {faiss_path}")
        print(f"📁 Index exists: {os.path.exists(faiss_path)}")

        # Check if faiss_index exists
        if not os.path.exists(faiss_path):
            error_msg = "❌ No document found. Please upload a PDF first."
            print(error_msg)
            return error_msg

        print("✅ faiss_index directory exists")

        # Load the "Brain" from the disk
        vector_store = FAISS.load_local(
            faiss_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print("✅ Vector store loaded successfully")

        # Search the document
        Retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        print("✅ Retriever created")

        # Test retrieval
        docs = Retriever.invoke(query)
        print(f"📄 Retrieved {len(docs)} documents")

        if len(docs) == 0:
            return "I couldn't find relevant information in the document. Try asking a more specific question."

        # Show previews of retrieved docs
        for i, doc in enumerate(docs[:3]):  # Show first 3 only
            print(f"📄 Doc {i+1} preview: {doc.page_content[:150]}...")

        # Combine the retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create the prompt
        prompt = f"""You are a Study Assistant. Use the Context below to answer the Question.
If the answer isn't in the context, say 'I don't see that in your notes'.

Context: {context}

Question: {query}

Answer:"""

        print("✅ Prompt created, calling Gemini...")

        # Call Gemini directly
        response = gemini_model.generate_content(prompt)
        answer = response.text

        print(f"\n💬 AI Answer Preview: {answer[:200]}...")
        print(f"{'='*60}\n")

        return answer

    except FileNotFoundError as e:
        error_msg = "❌ Document not found. Please upload a PDF first."
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return f"I encountered an error while reading the document: {str(e)}"

# !!! THIS WAS THE MISSING LINE !!!
retriever_tool = retrieve_data

# --- 3. CHAT MODEL SETUP (Using Native Gemini) ---
chat_model = genai.GenerativeModel("gemini-2.5-flash")

def generate_query_or_respond(state: MessagesState):
    """Decide whether to retrieve data or just chat."""
    print(f"\n🤖 generate_query_or_respond called")
    print(f"📨 State messages: {state['messages']}")

    # Get the last message
    last_message = state["messages"][-1]

    # Simple logic: if it looks like a question about documents, use retriever
    question_keywords = ["what", "how", "explain", "summary", "tell me", "describe"]
    user_content = last_message.content.lower() if hasattr(last_message, 'content') else str(last_message).lower()

    if any(keyword in user_content for keyword in question_keywords):
        # Use the retriever
        answer = retrieve_data.invoke({"query": user_content})
        response = AIMessage(content=answer)
    else:
        # Just chat normally
        response_text = chat_model.generate_content(user_content).text
        response = AIMessage(content=response_text)

    print(f"💬 Response: {response}\n")
    return {"messages": [response]}

# --- 4. GRADER (SIMPLIFIED) ---
def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Check if retrieved document is relevant."""
    print("\n📊 GRADING DOCUMENTS...")

    question = state["messages"][0].content
    context = state["messages"][-1].content

    print(f"Question: {question}")
    print(f"Context preview: {context[:200]}...")

    # Create grading prompt
    grade_prompt = f"""You are a grader assessing relevance.
Document: {context[:1000]}
Question: {question}
Does the document contain keywords related to the question?
Reply ONLY with the word 'YES' or 'NO'."""

    # Run the grader
    response = chat_model.generate_content(grade_prompt).text
    print(f"Grader response: {response}")

    # Simple check
    if "YES" in response.upper():
        print("✅ Document is relevant -> generate_answer")
        return "generate_answer"
    else:
        print("❌ Document not relevant -> rewrite_question")
        return "rewrite_question"

# --- 5. REWRITE QUESTION ---
def rewrite_question(state: MessagesState) -> dict:
    """Rewrite the question if the document wasn't relevant."""
    print("\n✏️ REWRITING QUESTION...")

    question = state["messages"][0].content
    print(f"Original question: {question}")

    rewrite_prompt = f"""Rewrite this question to be clearer for a search engine:
Original: {question}
Improved Question:"""

    response = chat_model.generate_content(rewrite_prompt).text
    print(f"Rewritten question: {response}")

    return {"messages": [HumanMessage(content=response)]}

# --- 6. GENERATE ANSWER ---
def generate_answer(state: MessagesState):
    """Generate the final answer."""
    print("\n💡 GENERATING FINAL ANSWER...")

    question = state["messages"][0].content
    context = state["messages"][-1].content

    print(f"Question: {question}")
    print(f"Context preview: {context[:200]}...")

    answer_prompt = f"""You are a Study Assistant. Use the Context below to answer the Question.
If the answer isn't in the context, say 'I don't see that in your notes'.

Context: {context}

Question: {question}

Answer:"""

    response = chat_model.generate_content(answer_prompt).text
    print(f"Generated answer: {response}")

    return {"messages": [AIMessage(content=response)]}
