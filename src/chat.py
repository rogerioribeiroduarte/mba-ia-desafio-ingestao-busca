from search import search_prompt
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
import time
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv("LLM_API_KEY")

def main():
    # 1. Iniciar llm
    model = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai")

    # 2. Montar o chain completo
    chain = search_prompt | model | StrOutputParser()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    while True:
      texto = input("Entre com sua pergunta: ")
      if texto == "q":
         break
      resposta = chain.invoke(texto)
      print(resposta)

if __name__ == "__main__":
    main()
