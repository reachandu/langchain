import streamlit as st
import sqlite3
import uuid

from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    SystemMessage,
    HumanMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# database
conn=sqlite3.connect("chat_memory1.db", check_same_thread=False)
cursor=conn.cursor()

cursor.execute(
    """
        CREATE TABLE IF NOT EXISTS chat_history (
            session_id TEXT,
            role TEXT,
            content TEXT
        )
    """
)
conn.commit()

def save_message(session_id: str, role:str, content:str):
    cursor.execute(
        "INSERT INTo chat_history(session_id, role, content) values (?,?,?)",
        (session_id, role, content)
    )
    conn.commit()

def load_chat_history(session_id:str) -> List[BaseMessage]:
    cursor.execute(
        "SELECT role, content FROM chat_history WHERE session_id = ?",
        (session_id,)
    )
    rows=cursor.fetchall()

    history: List[BaseMessage] = []
    for role, content in rows:
        if role=="human":
            history.append(HumanMessage(content=content))
        elif role=="ai":
            history.append(AIMessage(content=content))
    return history

def get_all_sessions():
    cursor.execute(
        "SELECT DISTINCT session_id FROM chat_history ORDER BY rowid DESC"
    )
    return [row[0] for row in cursor.fetchall()]

# streamlit configurations
st.set_page_config(page_title="Conversational RAG", layout="wide")
st.title("Conversational RAG with memory")

# sidebar
st.sidebar.title("Chats")

if "session_id" not in st.session_state:
    st.session_state.session_id=str(uuid.uuid4())
    st.session_state.chat_history=[]

if st.sidebar.button("New Chat"):
    st.session_state.session_id=str(uuid.uuid4())
    st.session_state.chat_history=[]    

if "session_id" not in st.session_state:
    st.session_state.session_id=str(uuid.uuid4())
    st.session_state.chat_history=[]

st.sidebar.markdown("Previous Conversations")

for sid in get_all_sessions():
    if st.sidebar.button(sid[:8]):
        st.session_state.session_id=sid
        st.session_state.chat_history=load_chat_history(sid)

session_id=st.session_state.session_id

# Load and index PDF document
@st.cache_resource
def load_vectorstore():
    loader=PyPDFLoader("sf_homeless_brief_final_web.pdf")
    document=loader.load()

    splitter=RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100
    )
    chunks=splitter.split_documents(documents=document)

    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-miniLM-L6-v2"
    )

    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vector_store

vectorstore=load_vectorstore()
retriever=vectorstore.as_retriever(search_kwargs={"k":4})

llm=ChatGroq(model="llama-3.3-70b-versatile")

prompt=ChatPromptTemplate(
    [
        SystemMessage(
            content=(
                """
                    You are a helpful AI assistant. Answer strictly from the provided context. If the answer is not present, just say you don't know.
                """
            )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "Context: {context}, question: {input}"   
        )
    ]
)

def conversational_rag(user_input:str, chat_history: List[BaseMessage]):
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

# Load chat history
if not st.session_state.chat_history:
    st.session_state.chat_history=load_chat_history(session_id)

# Chat window
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("AI").write(msg.content)

user_input=st.chat_input("Ask a question from the PDF:")

if user_input:
    st.chat_message("user").write(user_input)
    save_message(session_id, "human", user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    response, sources = conversational_rag(
        user_input, st.session_state.chat_history
    )

    st.chat_message("AI").write(response.content)
    save_message(session_id, "ai", response.content)
    st.session_state.chat_history.append(AIMessage(content=response.content))














