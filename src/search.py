from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
load_dotenv()

COLLECTION_NAME=os.getenv("PG_VECTOR_COLLECTION_NAME")
PGVECTOR_URL=os.getenv("DATABASE_URL")

os.environ['GOOGLE_API_KEY'] = os.getenv("LLM_API_KEY")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""
# 0. Preparar Prompt
prompt = PromptTemplate(input_variables=["contexto", "Pergunta"],template=PROMPT_TEMPLATE)

# 1. Preparar embedding
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 2. Abrir banco
engine = create_engine(PGVECTOR_URL)
db = PGVector(embeddings=embeddings,collection_name=COLLECTION_NAME,connection=engine,use_jsonb=True,)

# Função para fechar o pool de conexões do engine
def close_db_connection():
    print("\nFechando conexão com o banco de dados...")
    engine.dispose()
    print("Conexão fechada.")

def build_context(pergunta: str):
  relevant_items = db.similarity_search_with_score(pergunta, k=10)
  context = "\n".join([item.page_content for (item, score) in relevant_items])
  return context

def search_prompt(pergunta: str):
  return (
    {
        "pergunta": RunnablePassthrough(),
        "contexto": RunnableLambda(lambda x: build_context(x)),
    }
    | prompt
)

