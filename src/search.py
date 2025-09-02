from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, chain
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

@chain
def build_context(pergunta: str):
  # 1. Preparar embedding
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

  # 2. conectar no banco
  db = PGVector(embeddings=embeddings,collection_name=COLLECTION_NAME,connection=PGVECTOR_URL,use_jsonb=True,)

  # 3. fazer a busca vetorial
  relevant_items = db.similarity_search_with_score(pergunta, k=10)
  context = "\n".join([item.page_content for (item, score) in relevant_items])
  return {
    "pergunta": pergunta,
    "contexto": context
  }


# 5. Montar o chain de prompt que possa ser invocado para cada pergunta
def search_prompt(input):
  prompt = PromptTemplate(input_variables=["contexto", "Pergunta"],template=PROMPT_TEMPLATE)
  return build_context | prompt

