# LangGraph Proposal Processor

A document processing pipeline built with LangGraph that generates professional proposal responses using RAG (Retrieval Augmented Generation) with Supabase vector storage. Supports both OpenAI and Google Vertex AI as LLM providers.

## Overview

This tool helps you automatically generate proposal responses by leveraging your existing company documentation. It uses RAG to pull relevant information from your document store and generates customized sections for your proposal.

## Features

- RAG-powered document generation using Supabase vector store
- Choice of LLM providers (OpenAI GPT-4 or Google Vertex AI Gemini)
- Automated section generation:
  - Corporate Overview
  - Staff Profiles
  - Capabilities Relevant to Opportunity
  - Corporate Experiences
  - Responses to Opportunity Questions
- PDF generation
- Optional email delivery

## Prerequisites

1. Python 3.8+
2. Supabase account
3. Either OpenAI API key or Google Cloud Project with Vertex AI enabled
4. (Optional) SMTP server access for email delivery

## Setup

### 1. Install Dependencies

```bash
pip install langchain langgraph supabase reportlab google-cloud-aiplatform openai python-dotenv
```

### 2. Set Up Supabase Vector Store

1. Create a new Supabase project

2. Enable Vector Store by running these SQL commands in the Supabase SQL editor:

```sql
-- Enable the vector extension
create extension if not exists vector;

-- Create the documents table with vector storage
create table documents (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding vector(1536)
);

-- Create a function to handle vector similarity search
create or replace function match_documents (
  query_embedding vector(1536),
  match_count int
) returns table (id bigint, content text, metadata jsonb, similarity float)
language plpgsql
as $$
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

### 3. Environment Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 

Create a `.env` file:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# OpenAI (if using)
OPENAI_API_KEY=your_openai_key

# Vertex AI (if using)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GCP_PROJECT_ID=your_project_id
GCP_LOCATION=us-central1

# Email (optional)
SMTP_FROM=sender@example.com
SMTP_TO=recipient@example.com
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=your_email
SMTP_PASSWORD=your_app_password
```

## Usage

### 1. Upload Your Documents

First, upload your company documents to the vector store:

```python
from proposal_processor import upload_documents

upload_documents(
    docs_dir="path/to/your/docs",
    supabase_url="your_supabase_url",
    supabase_key="your_supabase_key",
    embedding_provider="openai",  # or "vertex"
    api_key="your_api_key"
)
```

### 2. Generate Proposals

#### Using OpenAI

```python
from proposal_processor import ProposalProcessor

processor = ProposalProcessor(
    supabase_url="your_supabase_url",
    supabase_key="your_supabase_key",
    llm_provider="openai",
    openai_api_key="your_openai_key"
)

# Configure email (optional)
processor.email_config = {
    "from": "sender@example.com",
    "to": "recipient@example.com",
    "smtp_server": "smtp.gmail.com",
    "username": "your_email",
    "password": "your_app_password"
}

# Run the processor
graph = processor.build_graph()
state = {"send_email": True}  # Set to False to skip email
graph.run(state)
```

#### Using Vertex AI

```python
processor = ProposalProcessor(
    supabase_url="your_supabase_url",
    supabase_key="your_supabase_key",
    llm_provider="vertex",
    project_id="your_gcp_project_id",
    location="us-central1",
    credentials_path="/path/to/credentials.json"
)

# Rest of the setup is the same as OpenAI example
```

## Output Structure

The processor generates a PDF with the following sections:

1. **Corporate Overview**: Company history, mission, and values
2. **Staff Profiles**: Key team members and their qualifications
3. **Capabilities**: Relevant competencies for the opportunity
4. **Corporate Experiences**: Past projects and success stories
5. **Opportunity Responses**: Direct answers to RFP questions

Each section is generated using RAG with relevant documents from your Supabase vector store.

## Document Types

The processor works best with the following types of documents:

1. Company overview documents
2. Employee resumes and bios
3. Past project summaries
4. Technical capability statements
5. Previous proposal responses
6. Marketing materials
7. Case studies

## Best Practices

1. **Document Organization**: Keep your documents well-organized in your repository
2. **Regular Updates**: Keep your vector store updated with latest company information
3. **Document Quality**: Ensure uploaded documents are clean and well-formatted
4. **Metadata**: Use descriptive filenames and maintain good metadata
5. **Testing**: Test the output with smaller document sets before scaling up

## Troubleshooting

Common issues and solutions:

1. **Vector Store Connection**: Ensure your Supabase credentials are correct
2. **Document Upload**: Check file permissions and formats
3. **LLM Errors**: Verify API keys and credentials
4. **PDF Generation**: Ensure sufficient disk space and permissions
5. **Email Errors**: Check SMTP settings and credentials

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.