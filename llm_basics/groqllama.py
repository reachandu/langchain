from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant")
response = llm.invoke("What is Langchain, explain in one sentence")
print(response) 
print("---------------------------------------------------")
print(response.content)