# Desafio MBA Engenharia de Software com IA - Full Cycle

Para execução do projeto é preciso seguir os seguintes passos
1. Ajustar a senha do banco de dados no docker-compose.yml
2. Subir o banco de dados:
<pre>
docker compose up -d
</pre>
3. Criar um arquivo .env com as chaves necessárias e senha do banco de dados, conforme o modelo em .env.example (não esquecer de ajustar a senha do banco)
4. Instalar as dependências
<pre>
pip install -r requirements.txt
</pre>
5. Executar ingestão do PDF:
<pre>
python src/ingest.py
</pre>
6. Rodar o chat:
<pre>
python src/chat.py
</pre>