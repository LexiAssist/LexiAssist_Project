from langchain_community.document_loaders import PyPDFLoader

def load_and_split_pdf(file_path):
    """
    Loads a PDF from a specific path and returns the documents.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
