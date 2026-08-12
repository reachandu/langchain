from dotenv import load_dotenv
load_dotenv()

# 1. load PDF document
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("sf_homeless_brief_final_web.pdf")
documents=loader.load()


# 2. Split documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs=text_splitter.split_documents(documents)


# 3. Create embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-miniLM-L6-v2"
)


# 4. Store embeddings in vector database
from langchain_chroma import Chroma

vector_store=Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="rag_collection"
)


# 5. retrievers

retriever=vector_store.as_retriever(
    search_type="mmr",
    searchkwargs={"k":2, "lambda_mult": 0.5}
)


# 6. Initialize LLM
from langchain_groq import ChatGroq

llm=ChatGroq(model_name="llama-3.3-70b-versatile")


# 7. Create Prompt template
from langchain_core.prompts import PromptTemplate

prompt=PromptTemplate(
    template="""
        You are an AI assistant. Use the following context to answer the question. If the naswer is not present in the context, say you don't know.
        Context: {context}
        question: {question}
""",
input_variables=["context", "question"]
)


# 8. Output parser definition
from langchain_core.output_parsers import StrOutputParser

parser=StrOutputParser()


# 9. Build RAG chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain=(
    {
        "context": retriever | format_docs,
        "question": lambda x: x
    } | prompt | llm | parser
)


# 10. ask question
query="Why homes are not cheap in golden state?"
response=rag_chain.invoke(query)

print(response)



