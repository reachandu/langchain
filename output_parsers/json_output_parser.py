from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model_name = "llama-3.3-70b-versatile"
model=ChatGroq(model=model_name)

parser = JsonOutputParser()

# prompt template
prompt_template = PromptTemplate(
    template="""
        Give me the name, age and city of a fictional character who is an american super hero. City has to be in USA. 
        {format_instructions}
    """,
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt_template | model | parser

response = chain.invoke({})
print("---------------------------------------------------")
print("Model:", model_name)
print("Response:", response)

