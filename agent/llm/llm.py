import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


client = ChatOllama(
    model="gpt-oss:120b",
    temperature=0.3,
    base_url=os.environ["OLLAMA_URL"],
    client_kwargs={
        "headers": {
         "Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"   
        }
    }
)