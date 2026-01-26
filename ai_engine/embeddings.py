from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from django.conf import settings

# Initialize the model once (it takes time to load)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(documents):
    """
    Takes raw documents, splits them, and creates a searchable vector store.
    """
    # 1. Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    doc_splits = text_splitter.split_documents(documents)

    # 2. Define the FAISS path (in project root)
    faiss_path = os.path.join(settings.BASE_DIR, "faiss_index")
    print(f"📁 Will save FAISS index to: {faiss_path}")

    # 3. Create the Vector Store
    vector_store = FAISS.from_documents(doc_splits, embedding_model)

    # 4. Save it to disk
    vector_store.save_local(faiss_path)
    print(f"✅ FAISS index saved to: {faiss_path}")
    print(f"✅ Verifying - Index exists: {os.path.exists(faiss_path)}")

    return vector_store
