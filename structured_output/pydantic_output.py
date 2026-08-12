from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model_name = "llama-3.3-70b-versatile"
model=ChatGroq(model=model_name)

# schema
class Schema(TypedDict):
    key_themes: list[str]=Field(description="List of key themes in the review") 
    summary: str=Field(description="Summary of the review")
    sentiment: Literal["positive", "negative", "neutral"]=Field(description="Sentiment of the review")
    pros: Optional[list[str]]=Field(default=None, description="List of pros mentioned in the review")
    cons: Optional[list[str]]=Field(default=None, description="List of cons mentioned in the review")
    name: Optional[str]=Field(default=None, description="Name of the product being reviewed")

structured_model = model.with_structured_output(Schema, strict=True)

prompt = """
    I have been using the Franklin X-40 pickleballs for several weeks, and overall, they have been a great choice for outdoor pickleball games. 
    The balls provide a consistent bounce, stable flight, good durability, and excellent visibility, making them suitable for both recreational and competitive players. 
    The quality feels premium, and they maintain their performance well during regular play. However, they can become slightly softer after extended use, may crack on very rough outdoor surfaces, 
    and are a little more expensive compared to some generic pickleballs. Despite these minor drawbacks, the Franklin X-40 offers a great balance of speed, control, 
    and reliability, making it one of the better options for outdoor pickleball.
"""

response_structured = structured_model.invoke(prompt)
print("---------------------------------------------------")
print("Model:", model_name)
print("Review:", prompt)
print("Structured Output:", response_structured)    