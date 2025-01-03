from typing import TypedDict, Optional,List
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import ChatOpenAI
from typing import Dict, Optional
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings, ChatVertexAI
from supabase import create_client
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from google.cloud import aiplatform
from absl import logging as absl_logging
import smtplib, time, os, logging, grpc

os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
absl_logging.set_verbosity(absl_logging.INFO)

def init_grpc():
    try:
        channel = grpc.insecure_channel('localhost:50051')
        grpc.channel_ready_future(channel).result(timeout=10)
    except grpc.FutureTimeoutError:
        pass
    except Exception as e:
        logger.warning(f"gRPC initialization warning: {str(e)}")

init_grpc()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        email_config: Optional[TypedDict] = None
    ):
        logger.info("Initializing ProposalProcessor with provider: %s", llm_provider)
        self.wait_between_api_calls = 1
        self.wait_between_api_sections= 60

        # Initialize LLM based on provider
        if llm_provider == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key required when using OpenAI provider")
            logger.info("Setting up OpenAI LLM")
            self.llm = ChatOpenAI(
                model_name="gpt-4-turbo-preview",
                api_key=openai_api_key,
                embeddings = OpenAIEmbeddings(openai_api_key)
            )
        elif llm_provider == "vertex":
            if not all([gcp_project_id, gcp_location, credentials_path]):
                raise ValueError("gcp_project_id, gcp_location, and credentials_path required for Vertex AI")
            logger.info("Setting up Vertex AI LLM")
            self.llm = ChatVertexAI(
                model_name="gemini-1.5-flash-002",
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

        logger.info("Setting up Supabase client and vector store")
        self.supabase = create_client(supabase_url, supabase_key)
        self.vectorstore = SupabaseVectorStore(
            client=self.supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )

        self.retriever = (self.vectorstore).as_retriever(search_kwargs={"k": 1})

    def retrieve_opportunity_docs(self, state: TypedDict) -> TypedDict:
        logger.info("Retrieving opportunity documents")
        docs = self.retriever.get_relevant_documents(
            "opportunity requirements scope objectives criteria"
        )
        logger.info("Retrieved %d opportunity documents", len(docs))
        state["opportunity_docs"] = docs
        return state

    def retrieve_corporate_docs(self, state: TypedDict) -> TypedDict:
        logger.info("Retrieving corporate documents")
        time.sleep(self.wait_between_api_calls)
        docs = self.retriever.get_relevant_documents(
            "company overview history mission values"
        )
        logger.info("Retrieved %d corporate documents", len(docs))
        state["corporate_docs"] = docs
        return state

    def retrieve_staff_docs(self, state: TypedDict) -> TypedDict:
        logger.info("Retrieving staff documents")
        time.sleep(self.wait_between_api_calls)
        docs = self.retriever.get_relevant_documents(
            "staff profiles expertise qualifications experience"
        )
        logger.info("Retrieved %d staff documents", len(docs))
        state["staff_docs"] = docs
        return state

    def retrieve_capabilities_docs(self, state: TypedDict) -> TypedDict:
        logger.info("Retrieving capabilities documents")
        time.sleep(self.wait_between_api_calls)
        opportunity_text = "\n".join([doc.page_content for doc in state["opportunity_docs"]])
        query = f"capabilities and competencies relevant to: {opportunity_text}"
        docs = self.retriever.get_relevant_documents(query)
        logger.info("Retrieved %d capabilities documents", len(docs))
        state["capabilities_docs"] = docs
        return state

    def retrieve_experience_docs(self, state: TypedDict) -> TypedDict:
        logger.info("Retrieving experience documents")
        time.sleep(self.wait_between_api_calls)
        opportunity_text = "\n".join([doc.page_content for doc in state["opportunity_docs"]])
        query = f"past projects and experience relevant to: {opportunity_text}"
        docs = self.retriever.get_relevant_documents(query)
        logger.info("Retrieved %d experience documents", len(docs))
        state["experience_docs"] = docs
        return state
    
    def export_debug_section(section: str, content: any, output_dir: str = "debug_output"):
        """Export section content to a text file for debugging."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{section.lower().replace(' ', '_')}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            if isinstance(content, list):
                f.write("\n".join([doc.page_content for doc in content]))
            else:
                f.write(str(content))
        logger.info(f"Exported debug file: {filepath}")

    def generate_section(self, state: TypedDict, section: str, docs_key: str) -> str:
        logger.info("Generating section: %s", section)
        templates = {
            "corporate_overview": "Write a comprehensive corporate overview based on: {documents}"
            #,"staff_profile": "Create detailed staff profiles highlighting relevant expertise based on: {documents}",
            #"unique_valuation_propositions": "Identify and explain why Northramp is best for this opportunity based on: {documents}",
            #"capabilities": "Describe capabilities relevant to the opportunity requirements based on: {documents}",
            #"experience": "Detail relevant corporate experience and past projects based on: {documents}",
            #"past_performance": "Summarize past performance and achievements based on: {documents}",
            #"responses": "Provide specific responses to opportunity questions and requirements based on: {documents}"
        }
        
        prompt = PromptTemplate(
            template=templates[section],
            input_variables=["documents"]
        )
        docs_text = "\n".join([doc.page_content for doc in state[docs_key]])
        chain = prompt | self.llm
        response = chain.invoke({"documents": docs_text})
        logger.info(response.content[:10])
        return response.content

    def build_document(self, state: TypedDict) -> TypedDict:
        logger.info("Building document")
        sections = {
            "corporate_overview": "corporate_docs"
            #,"staff_profile": "staff_docs",
            #"capabilities": "capabilities_docs",
            #"experience": "experience_docs",
            #"responses": "opportunity_docs"
        }
        
        content = {}
        for section, docs_key in sections.items():
            logger.info("Generating section: %s", section)
            content[section] = self.generate_section(state, section, docs_key)            
            time.sleep(self.wait_between_api_sections)
            
        pdf_path = "proposal_response.pdf"
        logger.info("Creating PDF: %s", pdf_path)
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica", 12)
        width, height = letter
        y = height - 50
        
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
            y -= 20
            if y < 50:  # Add new page if content exceeds one page
                c.showPage()
                y = height - 50
        
        c.save()
        logger.info("PDF created successfully")
        state["pdf_path"] = pdf_path
        return state

    def send_email(self, state: TypedDict) -> TypedDict:
        if not self.email_config or not state.get("send_email"):
            logger.info("Skipping email send (not configured or not requested)")
            return state

        logger.info("Preparing email")
        msg = MIMEMultipart()
        msg["From"] = self.email_config["from"]
        msg["To"] = self.email_config["to"]
        msg["Subject"] = "Proposal Response"

        with open(state["pdf_path"], "rb") as f:
            pdf = MIMEApplication(f.read(), _subtype="pdf")
            pdf.add_header("Content-Disposition", "attachment", filename="proposal_response.pdf")
            msg.attach(pdf)

        logger.info("Sending email to %s", self.email_config["to"])
        with smtplib.SMTP(self.email_config["smtp_server"]) as server:
            server.starttls()
            server.login(self.email_config["username"], self.email_config["password"])
            server.send_message(msg)
        logger.info("Email sent successfully")

        return state
    
    class ProposalState(TypedDict):
        opportunity_docs: List
        corporate_docs: List
        staff_docs: List
        capabilities_docs: List
        experience_docs: List
        pdf_path: str
        send_email: bool

    def build_graph(self) -> StateGraph:
        logger.info("Building workflow graph")
        workflow = StateGraph(self.ProposalState)

        workflow.add_node("opportunity", self.retrieve_opportunity_docs)
        workflow.add_node("corporate", self.retrieve_corporate_docs)
        # workflow.add_node("staff", self.retrieve_staff_docs)
        # workflow.add_node("capabilities", self.retrieve_capabilities_docs)
        # workflow.add_node("experience", self.retrieve_experience_docs)
        workflow.add_node("build", self.build_document)
        workflow.add_node("email", self.send_email)

        workflow.add_edge(START, "opportunity")
        workflow.add_edge("opportunity", "corporate")
        workflow.add_edge("corporate", "build")
        # workflow.add_edge("corporate", "staff")
        # workflow.add_edge("staff", "capabilities")
        # workflow.add_edge("capabilities", "experience")
        # workflow.add_edge("experience", "build")
        workflow.add_edge("build", "email")
        workflow.add_edge("email", END)

        logger.info("Workflow graph built successfully")
        return workflow.compile()