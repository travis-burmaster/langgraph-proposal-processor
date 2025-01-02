import os
from dotenv import load_dotenv
from proposal_processor import ProposalProcessor
# Load environment variables
load_dotenv()

# Initialize the processor
processor = ProposalProcessor(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_SERVICE_KEY"),
    openai_api_key="your_openai_key",
    email_config={
        "from": "sender@example.com",
        "to": "recipient@example.com",
        "smtp_server": "smtp.gmail.com",
        "username": "your_email",
        "password": "your_app_password"
    }
)

# Build and run the graph
graph = processor.build_graph()
state = {"send_email": False}  # Set to False to skip email
graph.run(state)