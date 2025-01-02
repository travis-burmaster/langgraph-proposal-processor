# Initialize the processor
processor = ProposalProcessor(
    supabase_url="your_supabase_url",
    supabase_key="your_supabase_key",
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
state = {"send_email": True}  # Set to False to skip email
graph.run(state)