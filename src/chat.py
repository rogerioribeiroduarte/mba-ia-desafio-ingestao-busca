from search import search_prompt, close_db_connection
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv("LLM_API_KEY")

# 3. Iniciar llm
model = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai")

def main():
    chain = search_prompt | model | StrOutputParser()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    try:
       while True:
         texto = input("Entre com sua pergunta: ")
         if texto == "q":
             break
         resposta = chain.invoke(texto)
         print(resposta)
    finally:
        close_db_connection()

if __name__ == "__main__":
    main()
