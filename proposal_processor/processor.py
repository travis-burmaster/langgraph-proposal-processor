from typing import Dict, Optional
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI, ChatVertexAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from supabase import create_client
from reportlab.pdfgen import canvas
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from google.cloud import aiplatform
import smtplib

class ProposalProcessor:
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        llm_provider: str = "openai",
        openai_api_key: Optional[str] = None,
        gcp_project_id: Optional[str] = None,
        gcp_location: Optional[str] = None,
        credentials_path: Optional[str] = None,
        email_config: Optional[Dict] = None
    ):
        
        # Initialize LLM based on provider
        if llm_provider == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key required when using OpenAI provider")
            self.llm = ChatOpenAI(
                model_name="gpt-4-turbo-preview",
                api_key=openai_api_key,
                embeddings = OpenAIEmbeddings(openai_api_key)
            )
        elif llm_provider == "vertex":
            if not all([gcp_project_id, gcp_location, credentials_path]):
                raise ValueError("gcp_project_id, gcp_location, and credentials_path required for Vertex AI")
            # aiplatform.init(
            #     project=gcp_project_id,
            #     gcp_location=gcp_location,
            #     credentials=credentials_path
            # )
            self.llm = ChatVertexAI(
                model_name="gemini-1.5-flash",
                max_output_tokens=2048,
                temperature=0.2,
                project=gcp_project_id,
                location=gcp_location,
                credentials_path=credentials_path
            )
            embeddings = VertexAIEmbeddings(
                model_name="text-embedding-004",
                project=gcp_project_id,
                location=gcp_location
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
            
        self.email_config = email_config

        self.supabase = create_client(supabase_url, supabase_key)
        self.vectorstore = SupabaseVectorStore(
            client=self.supabase,
            embedding=embeddings,
            table_name="documents"
        )

        self.retriever = (self.vectorstore).as_retriever()

    def retrieve_opportunity_docs(self, state: Dict) -> Dict:
        docs = self.retriever.get_relevant_documents(
            "opportunity requirements scope objectives criteria"
        )
        state["opportunity_docs"] = docs
        return state

    def retrieve_corporate_docs(self, state: Dict) -> Dict:
        docs = self.retriever.get_relevant_documents(
            "company overview history mission values"
        )
        state["corporate_docs"] = docs
        return state

    def retrieve_staff_docs(self, state: Dict) -> Dict:
        docs = self.retriever.get_relevant_documents(
            "staff profiles expertise qualifications experience"
        )
        state["staff_docs"] = docs
        return state

    def retrieve_capabilities_docs(self, state: Dict) -> Dict:
        opportunity_text = "\n".join([doc.page_content for doc in state["opportunity_docs"]])
        query = f"capabilities and competencies relevant to: {opportunity_text}"
        docs = self.retriever.get_relevant_documents(query)
        state["capabilities_docs"] = docs
        return state

    def retrieve_experience_docs(self, state: Dict) -> Dict:
        opportunity_text = "\n".join([doc.page_content for doc in state["opportunity_docs"]])
        query = f"past projects and experience relevant to: {opportunity_text}"
        docs = self.retriever.get_relevant_documents(query)
        state["experience_docs"] = docs
        return state

    def generate_section(self, state: Dict, section: str, docs_key: str) -> str:
        templates = {
            "corporate_overview": "Write a comprehensive corporate overview based on: {documents}",
            "staff_profile": "Create detailed staff profiles highlighting relevant expertise based on: {documents}",
            "capabilities": "Describe capabilities relevant to the opportunity requirements based on: {documents}",
            "experience": "Detail relevant corporate experience and past projects based on: {documents}",
            "responses": "Provide specific responses to opportunity questions and requirements based on: {documents}"
        }
        
        prompt = PromptTemplate(
            template=templates[section],
            input_variables=["documents"]
        )
        docs_text = "\n".join([doc.page_content for doc in state[docs_key]])
        chain = prompt | self.llm
        return chain.invoke({"documents": docs_text})

    def build_document(self, state: Dict) -> Dict:
        sections = {
            "corporate_overview": "corporate_docs",
            "staff_profile": "staff_docs",
            "capabilities": "capabilities_docs",
            "experience": "experience_docs",
            "responses": "opportunity_docs"
        }
        
        content = {}
        for section, docs_key in sections.items():
            content[section] = self.generate_section(state, section, docs_key)
            
        pdf_path = "proposal_response.pdf"
        c = canvas.Canvas(pdf_path)
        y = 800
        
        for section, text in content.items():
            title = section.replace("_", " ").upper()
            c.drawString(50, y, title)
            y -= 20
            
            words = text.split()
            line = ""
            for word in words:
                if len(line + word) < 80:
                    line += word + " "
                else:
                    c.drawString(50, y, line)
                    y -= 15
                    line = word + " "
            if line:
                c.drawString(50, y, line)
            y -= 30
        
        c.save()
        state["pdf_path"] = pdf_path
        return state

    def send_email(self, state: Dict) -> Dict:
        if not self.email_config or not state.get("send_email"):
            return state

        msg = MIMEMultipart()
        msg["From"] = self.email_config["from"]
        msg["To"] = self.email_config["to"]
        msg["Subject"] = "Proposal Response"

        with open(state["pdf_path"], "rb") as f:
            pdf = MIMEApplication(f.read(), _subtype="pdf")
            pdf.add_header("Content-Disposition", "attachment", filename="proposal_response.pdf")
            msg.attach(pdf)

        with smtplib.SMTP(self.email_config["smtp_server"]) as server:
            server.starttls()
            server.login(self.email_config["username"], self.email_config["password"])
            server.send_message(msg)

        return state

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(state_schema={
            "opportunity_docs": list,
            "corporate_docs": list,
            "staff_docs": list,
            "capabilities_docs": list,
            "experience_docs": list,
            "pdf_path": str,
            "send_email": bool
        })

        workflow.add_node("opportunity", self.retrieve_opportunity_docs)
        workflow.add_node("corporate", self.retrieve_corporate_docs)
        workflow.add_node("staff", self.retrieve_staff_docs)
        workflow.add_node("capabilities", self.retrieve_capabilities_docs)
        workflow.add_node("experience", self.retrieve_experience_docs)
        workflow.add_node("build", self.build_document)
        workflow.add_node("email", self.send_email)

        workflow.add_edge("opportunity", "corporate")
        workflow.add_edge("corporate", "staff")
        workflow.add_edge("staff", "capabilities")
        workflow.add_edge("capabilities", "experience")
        workflow.add_edge("experience", "build")
        workflow.add_edge("build", "email")
        workflow.add_edge("email", END)

        return workflow