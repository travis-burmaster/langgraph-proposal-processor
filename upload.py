import os
from dotenv import load_dotenv
from proposal_processor import upload_documents

# Load environment variables
load_dotenv()

# Upload documents
upload_documents(
    docs_dir="docs",  # Directory containing your documents
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_KEY"),
    embedding_provider="openai",  # or "vertex"
    api_key=os.getenv("OPENAI_API_KEY"),  # or None if using vertex
    project_id=os.getenv("GCP_PROJECT_ID"),  # Required for vertex
    location=os.getenv("GCP_LOCATION")  # Required for vertex
)