from langchain.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS

# --- IMPORT EMBEDDINGS CORRECTLY ---
from ai_engine.embeddings import embedding_model

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# --- 1. SETUP LLM ---
repo_id = "mistralai/Mistral-Nemo-Base-2407"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    temperature=0.3,
    max_new_tokens=512,
    timeout=180,  # 3 minute timeout for slow responses
)

llm_chain = llm


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

        # Create QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_chain,
            chain_type="stuff",
            retriever=Retriever,
            return_source_documents=False
        )
        print("✅ QA Chain created")

        # Get the answer
        print("🤖 Invoking QA chain...")
        result = qa_chain.invoke({"query": query})

        # Handle different result formats
        if isinstance(result, dict):
            answer = result.get('result', str(result))
        else:
            answer = str(result)

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

# --- 3. CHAT MODEL SETUP ---
chat_model = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="conversational",
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",
)
chat = ChatHuggingFace(llm=chat_model, temperature=0)

def generate_query_or_respond(state: MessagesState):
    """Decide whether to retrieve data or just chat."""
    print(f"\n🤖 generate_query_or_respond called")
    print(f"📨 State messages: {state['messages']}")

    # We bind the tool so the LLM knows it exists
    response = chat.bind_tools([retriever_tool]).invoke(state["messages"])

    print(f"💬 LLM Response: {response}\n")
    return {"messages": [response]}

# --- 4. GRADER (SIMPLIFIED) ---
# We use a simple prompt instead of a complex agent to avoid crashes
GRADE_PROMPT = PromptTemplate.from_template(
    "You are a grader assessing relevance.\n"
    "Document: {context}\n"
    "Question: {question}\n"
    "Does the document contain keywords related to the question? \n"
    "Reply ONLY with the word 'YES' or 'NO'."
)

grade_chain = GRADE_PROMPT | chat_model

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Check if retrieved document is relevant."""
    print("\n📊 GRADING DOCUMENTS...")

    question = state["messages"][0].content
    context = state["messages"][-1].content

    print(f"Question: {question}")
    print(f"Context preview: {context[:200]}...")

    # Run the grader
    response = grade_chain.invoke({"question": question, "context": context})
    print(f"Grader response: {response}")

    # Simple check
    if "YES" in response.upper():
        print("✅ Document is relevant -> generate_answer")
        return "generate_answer"
    else:
        print("❌ Document not relevant -> rewrite_question")
        return "rewrite_question"

# --- 5. REWRITE QUESTION ---
REWRITE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    "Rewrite this question to be clearer for a search engine:\n"
    "Original: {question}\n"
    "Improved Question:"
)
rewrite_chain = REWRITE_PROMPT_TEMPLATE | chat_model

def rewrite_question(state: MessagesState) -> dict:
    """Rewrite the question if the document wasn't relevant."""
    print("\n✏️ REWRITING QUESTION...")

    question = state["messages"][0].content
    print(f"Original question: {question}")

    response = rewrite_chain.invoke({"question": question})
    print(f"Rewritten question: {response}")

    return {"messages": [HumanMessage(content=response)]}

# --- 6. GENERATE ANSWER ---
GENERATE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    "You are a Study Assistant. Use the Context below to answer the Question.\n"
    "If the answer isn't in the context, say 'I don't see that in your notes'.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n"
    "Answer:"
)
generate_chain = GENERATE_PROMPT_TEMPLATE | chat_model

def generate_answer(state: MessagesState):
    """Generate the final answer."""
    print("\n💡 GENERATING FINAL ANSWER...")

    question = state["messages"][0].content
    context = state["messages"][-1].content

    print(f"Question: {question}")
    print(f"Context preview: {context[:200]}...")

    response = generate_chain.invoke({"question": question, "context": context})
    print(f"Generated answer: {response}")

    return {"messages": [response]}
