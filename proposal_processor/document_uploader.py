from pathlib import Path
from typing import Literal
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

def upload_documents(
    docs_dir: str,
    supabase_url: str,
    supabase_key: str,
    embedding_provider: Literal["openai", "vertex"] = "openai",
    api_key: str = None,
    project_id: str = None,
    location: str = None
):
    supabase = create_client(supabase_url, supabase_key)
    
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
    
    docs_path = Path(docs_dir)
    for doc_path in docs_path.glob("**/*"):
        if doc_path.is_file():
            with open(doc_path, 'r') as f:
                content = f.read()
                metadata = {
                    "source": str(doc_path),
                    "filename": doc_path.name
                }
                vectorstore.add_texts([content], metadatas=[metadata])