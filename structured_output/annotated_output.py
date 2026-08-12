from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

model_name = "llama-3.3-70b-versatile"
model=ChatGroq(model=model_name)

# schema
class Schema(TypedDict):
    key_themes: Annotated[list[str], "List of key themes in the review"]
    summary: Annotated[str, "Summary of the review"]
    sentiment: Annotated[str, "Sentiment of the review"]
    pros: Annotated[list[str], "List of pros mentioned in the review"]
    cons: Annotated[list[str], "List of cons mentioned in the review"]

structured_model = model.with_structured_output(Schema, method="function_calling")
prompt = """
    Analyze the following product review and provide a structured output with the following fields:
    - key_themes: List of key themes in the review
    - summary: Summary of the review
    - sentiment: Sentiment of the review (positive, negative, or neutral)
    - pros: List of pros mentioned in the review
    - cons: List of cons mentioned in the review

    Review:
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
