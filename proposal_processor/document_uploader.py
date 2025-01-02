import os
import io
from typing import List
from langchain.schema import Document
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

def upload_documents(directory_path: str) -> None:
    """
    Upload documents from a directory to Chroma vector store.
    
    # Configure embeddings based on provider
    if embedding_provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key required")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    elif embedding_provider == "vertex":
        if not all([project_id, location]):
            raise ValueError("project_id and location required for Vertex AI")
        embeddings = VertexAIEmbeddings(
            model_name="textembedding-gecko",
            project=project_id,
            location=location
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
    
    vectorstore = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents"
    )
    
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    
    vectordb.persist()