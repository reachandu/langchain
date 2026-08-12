from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model_name = "llama-3.3-70b-versatile"
model=ChatGroq(model=model_name)

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(gt=18, lt=60, description="The age of the person")
    city: str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

# prompt template
prompt_template = PromptTemplate(
    template="""
        Give me the name, age and city of a fictional character who is a fictional character from {country}. City has to be in the {country}. 
        Return the response in the following format:        
        {format_instructions}
    """,
    input_variables=["country"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = prompt_template.format_prompt(country="UK")
chain = prompt_template | model | parser

response = chain.invoke({"country": "Mexico"})
print("---------------------------------------------------")
print("Model:", model_name)
print("Response:", response)

