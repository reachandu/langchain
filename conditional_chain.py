from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model_name = "llama-3.3-70b-versatile"
model=ChatGroq(model=model_name)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The feedback provided by the user. It can be either 'positive' or 'negative'."
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    input_variables=["feedback"],
    template="""
        classify the sentiment of the following text into positive or negative: {feedback}
        {format_instruction}
    """,
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    input_variables=["feedback"],
    template="""
        Write an appropriate response to this positive feedback: {feedback}
    """
)   

prompt3 = PromptTemplate(
    input_variables=["feedback"],
    template="""
        Write an appropriate response to this negative feedback: {feedback}
    """
)   

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),    
    RunnableLambda(lambda x: "Invalid sentiment value. Please provide either 'positive' or 'negative' feedback.")
)

chain = classifier_chain | branch_chain
result = chain.invoke({"feedback": "I love this product! It has changed my life for the better."})
print("---------------------------------------------------")
print("Model:", model_name)
print("Result:", result)

result2 = chain.invoke({"feedback": "This product is terrible. It broke after one use and the customer service was unhelpful."})
print("---------------------------------------------------")
print("Model:", model_name)
print("Result:", result2)
