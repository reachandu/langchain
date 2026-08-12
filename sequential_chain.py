from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model_name = "llama-3.3-70b-versatile"
model=ChatGroq(model=model_name)

# first prompt template
prompt_template_1 = PromptTemplate(
    input_variables=["topic"],
    template="""
        Generate a detailed report on {topic}.
    """
)

# second prompt template
prompt_template_2 = PromptTemplate(
    input_variables=["text"],
    template="""
        Generate a three point summary of the following text: {text}
    """
)

parser = StrOutputParser()

chain = prompt_template_1 | model | parser | prompt_template_2 | model | parser

response = chain.invoke({"topic": "Labrador Retriever"})
print("---------------------------------------------------")
print("Model:", model_name)
print("Response:", response)

