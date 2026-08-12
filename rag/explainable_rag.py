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


# 5. similarity search
query="What is this document about?"
retrieved_docs=vector_store.similarity_search(
    query=query,
    k=3
)


# 6. build context plus source
context_text=""
sources=[]

for i, doc in enumerate(retrieved_docs):
    page=doc.metadata.get('page', 'N/A')
    source=doc.metadata.get('source', "PDF")
    context_text += f"\n Chunk {i+1}: \n{doc.page_content}\n"
    sources.append({
        "chunk": i+1,
        "page": page,
        "source": sources,
        "content": doc.page_content
    })


# 7. Prompt LLM with context
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm=ChatGroq(
    model="llama-3.3-70b-versatile"
)

prompt=PromptTemplate(
    template="""
        Answer the question only using the context below. Also provide the actual chunk during retrieval along with page number where I can find the chunk

        context: {context}
        question: {question}
    """,
    input_variables=['context', 'question']
)

parser=StrOutputParser()

chain = prompt| llm | parser

answer=chain.invoke({
    "context": context_text,
    "question": query
})

print(answer)

for src in sources:
    print(f"Chunk: {src['chunk']} | Page: {src['page']}")
    print(src['content'])
    print('-----------------------------------')

