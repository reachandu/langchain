from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model_name = "llama-3.1-8b-instant"
model=ChatGroq(model=model_name)

# schema
class Review(TypedDict):
    summary: str
    sentiment: str

review = "This product X-40 pickleballs is amazing! The quality is top-notch, and they last much longer than any other pickleballs I've used. Highly recommend to anyone looking for durable and high-performance pickleballs."

prompt = """
    You are a helpful assistant that summarizes product reviews and provides sentiment analysis.
    Please provide a summary and sentiment (positive, negative, or neutral) for the following review: 
    {review}
    """

response = model.invoke(prompt.format(review=review))
print("---------------------------------------------------")
print("Model:", model_name)
print("Review:", review)
print("Response:", response.content)
print("---------------------------------------------------")


structured_model = model.with_structured_output(Review)
response_structured = structured_model.invoke(prompt.format(review=review))
print("---------------------------------------------------")
print("Structured Output:", response_structured)
