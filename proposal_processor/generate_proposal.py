import os
from dotenv import load_dotenv
from proposal_processor import ProposalProcessor

# Load environment variables
load_dotenv()

# Initialize processor with OpenAI
processor = ProposalProcessor(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_KEY"),
    llm_provider="openai",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Optional: Configure email
if os.getenv("SMTP_FROM"):
    processor.email_config = {
        "from": os.getenv("SMTP_FROM"),
        "to": os.getenv("SMTP_TO"),
        "smtp_server": os.getenv("SMTP_SERVER"),
        "username": os.getenv("SMTP_USERNAME"),
        "password": os.getenv("SMTP_PASSWORD")
    }

# Run the processor
graph = processor.build_graph()
state = {"send_email": True}  # Set to False to skip email
graph.run(state)