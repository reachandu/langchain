from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

print("Blog Post Generator")
print("---------------------------------------------------")
print("Provide a topic for the blog post, type exit to quit.")
topic = input("Topic: ")
if topic.lower() == "exit":
    exit()

chat_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("You are a professional blogger. Help generate engaging, informative, and well-structured content about {topic}."),
        HumanMessagePromptTemplate.from_template("Write a detailed blog post about {topic}?")
    ]
)

# Initiate chat history with the system message
chat_history = []

while True:
    prompt = chat_prompt.format_prompt(topic=topic)
    chat_history.extend(prompt.to_messages())
    
    response = llm.invoke(chat_history)
    print("---------------------------------------------------")
    print("Generated Blog Post:")
    print(response.content[0].get("text"))
    
    print("---------------------------------------------------")
    print("Would you like to generate another blog post? (yes/no)")
    choice = input().strip().lower()
    if choice != "yes":
        break
    print("Provide a topic for the blog post, type exit to quit.")
    topic = input("Topic: ")
    if topic.lower() == "exit":
        exit()        
