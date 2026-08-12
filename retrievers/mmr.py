import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

documents=[
    Document(page_content="Langchain makes it easy to work with LLMs"),
    Document(page_content="Langchain helps developer build LLM applications easily"),
    Document(page_content="Chroma is a vector database optimised for LLM based search"),
    Document(page_content="Embeddings convert text into high-dimentional vectors"),
    Document(page_content="MMR helps you get diverse results when doing similar search"),
    Document(page_content="OpenAI provides powerful embedding models")    
]

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore=FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3, "lambda_mult": 0.25}
)

query="What is Langchain?"
results=retriever.invoke(query)

print(results)
