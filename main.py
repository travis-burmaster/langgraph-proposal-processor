import os
from dotenv import load_dotenv
from proposal_processor import ProposalProcessor
from google.cloud import aiplatform

# Load environment variables
load_dotenv()

# Initialize VertexAI
if os.getenv("USE_VERTEX_AI", "false").lower() == "true":
    aiplatform.init(
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    )

# Initialize the processor
processor = ProposalProcessor(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_SERVICE_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    use_vertex_ai=os.getenv("USE_VERTEX_AI", "false").lower() == "true",
    vertex_ai_model="text-embedding-004",
    email_config={
        "from": os.getenv("EMAIL_FROM"),
        "to": os.getenv("EMAIL_TO"),
        "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        "username": os.getenv("EMAIL_USERNAME"),
        "password": os.getenv("EMAIL_PASSWORD")
    }
)

# Build and run the graph
graph = processor.build_graph()
state = {"send_email": False}  # Set to False to skip email
graph.run(state)