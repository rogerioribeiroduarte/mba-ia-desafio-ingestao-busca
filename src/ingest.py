from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
COLLECTION_NAME=os.getenv("PG_VECTOR_COLLECTION_NAME")
PGVECTOR_URL=os.getenv("DATABASE_URL")
os.environ['GOOGLE_API_KEY'] = os.getenv("LLM_API_KEY")

def ingest_pdf():
    # Inicializar embedding
	embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Inicializar banco
	db = PGVector(embeddings=embeddings,collection_name=COLLECTION_NAME,connection=PGVECTOR_URL,use_jsonb=True,)
    # Carregar arquivo
	loader = PyPDFLoader(PDF_PATH)
	docs = loader.load()
    # Fazer split
	splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150,)
	chunks = splitter.split_documents(docs)
	# Enriquecer splits
	enriched = [
	    Document(
		    page_content=d.page_content,
            	metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
    	)
		for d in chunks
    ]  
	ids = [f"doc-{i}" for i in range(len(enriched))]
	# Inserir conteúdo no banco vetorial
	db.add_documents(documents=enriched, ids=ids)

if __name__ == "__main__":
    ingest_pdf()