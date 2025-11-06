# file_handler.py
"""
Handles file uploads, reading content, chunking documents,
and embedding them into ChromaDB.
"""

import os
import tempfile
import pandas as pd
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader, CSVLoader, JSONLoader, UnstructuredWordDocumentLoader
)
from chroma_store import get_vector_store

def load_document(file):
    """
    Load and parse a file depending on its type.

    Args:
        file (UploadedFile): Streamlit uploaded file.

    Returns:
        list: A list of documents/texts extracted from the file.
    """
    suffix = file.name.split(".")[-1].lower()
    temp_path = os.path.join(tempfile.gettempdir(), file.name)

    # Save uploaded file temporarily
    with open(temp_path, "wb") as f:
        f.write(file.read())

    # Load based on file type
    if suffix == "pdf":
        return PyPDFLoader(temp_path).load()
    elif suffix in ["xls", "xlsx"]:
        df = pd.read_excel(temp_path)
        return [{"page_content": df.to_string()}]
    elif suffix == "csv":
        return CSVLoader(temp_path).load()
    elif suffix == "json":
        return JSONLoader(temp_path).load()
    elif suffix in ["doc", "docx"]:
        return UnstructuredWordDocumentLoader(temp_path).load()
    else:
        st.warning("⚠️ Unsupported file type.")
        return []

def chunk_and_embed(docs, api_key):
    """
    Chunk text documents and embed them into ChromaDB.

    Args:
        docs (list): Loaded documents.
        api_key (str): OpenAI API key for generating embeddings.

    Returns:
        int: Number of chunks successfully embedded.
    """
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = get_vector_store()

    # Store the chunks in ChromaDB
    vectordb.add_documents(chunks)
    vectordb.persist()

    return len(chunks)
