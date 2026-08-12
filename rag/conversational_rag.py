from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from typing import List
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

loader=PyPDFLoader("sf_homeless_brief_final_web.pdf")
documents=loader.load()

llm=ChatGroq(model="llama-3.3-70b-versatile")

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks=splitter.split_documents(documents)

embeddings=HuggingFaceEmbeddings(model="sentence-transformers/all-miniLM-L6-v2")

vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever=vector_store.as_retriever(search_kwargs={"k":4})

prompt=ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
                You are a helpful AI assistant. Answer strictly from the provided context. If the answer is not present in the context, just say you don't know.
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "Context: \n {context} \n\n question: \n{input}"
        )
    ]
)

def conversational_rag(user_input: str, chat_history: List[BaseMessage]):
    docs=retriever.invoke(user_input)
    context="\n\n".join(
        f"[Page {d.metadata.get('page', 'N/A')}]\n{d.page_content}" for d in docs
    )
    messages=prompt.invoke(
        {
            "input": user_input,
            "context": context,
            "chat_history": chat_history
        }
    )

    response=llm.invoke(messages)
    return response, docs

chat_history: List[BaseMessage]=[]
print("Conversational RAG")
while True:
    user_input=input("you: ")
    if user_input.lower() =="exit":
        break
    chat_history.append(HumanMessage(content=user_input))
    response, sources = conversational_rag(user_input, chat_history=chat_history)

    chat_history.append(AIMessage(content=response.content))
    print("\nAI:", response.content)
