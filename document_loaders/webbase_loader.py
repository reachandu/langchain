import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model_name = "llama-3.3-70b-versatile"
model=ChatGroq(model=model_name)

prompt_template = PromptTemplate(
    input_variables=["question", "text"],
    template="""
        Answer the following question {question} from the following text {text}. If the answer is not contained within the text below, say "I don't know".
    """
)

url = "https://kids.britannica.com/students/article/social-media/635756"
loader = WebBaseLoader(web_path=url)

loaded_documents = loader.load()

parser = StrOutputParser()

chain = prompt_template | model | parser

result = chain.invoke({"question": "How do you ensure your teens are safe on social media?", "text": loaded_documents[0].page_content})
# print("---------------------------------------------------")
# print(loaded_documents[0].page_content)
print("---------------------------------------------------")
print("Model:", model_name)
print("Response:", result)  
