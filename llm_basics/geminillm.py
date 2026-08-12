from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

response = llm.invoke("What is Langchain, explain in one sentence")
print(response)
print("---------------------------------------------------")
print(response.content[0].get("text"))