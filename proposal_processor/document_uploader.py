from pathlib import Path
from typing import Literal
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client
import PyPDF2
import os

def upload_documents(
    docs_dir: str,
    supabase_url: str,
    supabase_key: str,
    embedding_provider: Literal["openai", "vertex"] = "openai",
    api_key: str = None,
    project_id: str = None,
    location: str = None,
    debug_dir: str = "debug_output"
):
    supabase = create_client(supabase_url, supabase_key)
    
    if embedding_provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key required")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    elif embedding_provider == "vertex":
        if not all([project_id, location]):
            raise ValueError("project_id and location required for Vertex AI")
        embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004",
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
    
    # Create debug directory if it doesn't exist
    os.makedirs(debug_dir, exist_ok=True)
    
    docs_path = Path(docs_dir)
    for doc_path in docs_path.glob("**/*"):
        if doc_path.is_file():
            if doc_path.suffix.lower() == '.pdf':
                with open(doc_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ""
                    # Export each page separately
                    for i, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        content += page_text + "\n"
                        debug_path = Path(debug_dir) / f"{doc_path.stem}_page_{i+1}.txt"
                        with open(debug_path, 'w', encoding='utf-8') as debug_file:
                            debug_file.write(page_text)
            else:
                with open(doc_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    debug_path = Path(debug_dir) / f"{doc_path.stem}_content.txt"
                    with open(debug_path, 'w', encoding='utf-8') as debug_file:
                        debug_file.write(content)
            
            metadata = {
                "source": str(doc_path),
                "filename": doc_path.name
            }
            content = content.replace('\x00', '')
            
            # Export final processed content
            final_debug_path = Path(debug_dir) / f"{doc_path.stem}_final.txt"
            with open(final_debug_path, 'w', encoding='utf-8') as debug_file:
                debug_file.write(content)
                
            vectorstore.add_texts([content], metadatas=[metadata])