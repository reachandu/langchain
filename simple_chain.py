from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model_name = "llama-3.3-70b-versatile"
model=ChatGroq(model=model_name)

# prompt template
prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="""
        Please provide three facts about the topic {topic}.
    """
)

parser = StrOutputParser()

chain = prompt_template | model | parser 

response = chain.invoke({"topic": "Money"})
print("---------------------------------------------------")
print("Model:", model_name)
print("Response:", response)

