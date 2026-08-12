from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model_name = "llama-3.3-70b-versatile"
model=ChatGroq(model=model_name)

# schema
json_schema = {
    "type": "object",
    "title": "Review",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of key themes in the review"
        },
        "summary": {
            "type": "string",
            "description": "Summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "Sentiment of the review"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "List of pros mentioned in the review"
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "List of cons mentioned in the review"
        },
        "name": {
            "type": ["string", "null"],
            "description": "Name of the product being reviewed"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

structured_model = model.with_structured_output(json_schema, strict=True)

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