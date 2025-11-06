# chroma_store.py
"""
This module manages the initialization of a ChromaDB vector store
for storing and retrieving embedded document data.
"""

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def get_vector_store(persist_directory="chroma_db"):
    """
    Initialize and return a Chroma vector store instance.

    Args:
        persist_directory (str): The directory path to persist Chroma data.

    Returns:
        Chroma: A vector store instance.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore
