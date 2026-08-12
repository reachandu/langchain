import os
from dotenv import load_dotenv

from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    SystemMessage,
    HumanMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# 1. Load multiple documents
DOC_FOLDER = "docs"

all_documents: List[Document] = []

for file in os.listdir(DOC_FOLDER):
    if file.endswith('.pdf'):
        loader=PyPDFLoader(os.path.join(DOC_FOLDER, file))
        docs=loader.load()

        # attach metadata
        for d in docs:
            d.metadata['source_document']=file

        all_documents.extend(docs)
print(f"Loaded {len(all_documents)} pages from multiple documents")        

# 2. split documents
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks=splitter.split_documents(all_documents)
print(f"Total chunks created: {len(chunks)}")


# 3. Create embeddings
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-miniLM-L6-v2"
)


# 4. Vector Store
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="multi_doc_rag"
)
retriever=vectorstore.as_retriever(search_kwargs={"k":4})


# 5. Define LLM
llm=ChatGroq(
    model="llama-3.3-70b-versatile"
)


# 6. Prompt
prompt=ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """
                    You are an expert AI assistant. Answer only using the provided context. If multiple documents are involved, clearly mention them.
                    If the answer is not present, just say you don't know
                """
            )
        ),
        (
            "human",
            "Context: {context}, question: {question}"
        )
    ]
)


# 7. RAG function
def multi_document_rag(query: str):
    docs=retriever.invoke(query)
    context="\n\n".join(
        f"""
            [Document:{d.metadata.get('source_document')}]
            {d.page_content}
        """
        for d in docs
    )

    messages=prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )
    response=llm.invoke(messages)
    return response.content, docs


# 8. Run RAG application
print(f"Muti document RAG, type exit to quit")
while True:
    query=input("You: ")
    if query.lower() == 'exit':
        break;
    answer, sources=multi_document_rag(query)
    print("AI Answer: ", answer)

    for i, d in enumerate(sources):
        print(f"""{i+1}, {d.metadata.get('source_document')}""")





