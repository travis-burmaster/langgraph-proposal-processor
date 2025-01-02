import os
from proposal_processor import upload_documents
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

upload_documents(
    docs_dir=os.getenv("PATH_TO_FILES"),
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_SERVICE_KEY"),
    embedding_provider="vertex",  # or "vertex"
    api_key="your_api_key",
    project_id=os.getenv("GCP_PROJECT_ID"),  # Required for vertex
    location=os.getenv("GCP_LOCATION") 
)