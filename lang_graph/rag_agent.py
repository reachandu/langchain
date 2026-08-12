from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage, SystemMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.6-flash")
embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# PDF Loader
pdf_loader=PyPDFLoader("sf_homeless_brief_final_web.pdf")
pages=pdf_loader.load()

# Text Splitter
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,    
)
pages_split=text_splitter.split_documents(pages)
print(f"Text splitter={len(pages_split)}")

# Vector database
try:
    vectorstore=Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        collection_name='sf_homeless_collection'
    )
    print("Vecorstore created!")
except Exception as e:
    print(f"An error occured while creating vector store: {e}")

# Retriever
retriever=vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)    

@tool
def retriever_tool(query:str)-> str:
    """ This tool searches for query in the vector store and returns documents from the sf homeless PDF"""
    docs=retriever.invoke(query)
    if not docs:
        return "No relavant information found in the document"
    results=[]
    for i, doc in enumerate(docs):
        results.append(f"Source {i+1}:\n{doc.page_content}\n")
    return "\n".join(results)

tool=[retriever_tool]
llm=model.bind_tools(tool)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state:AgentState):
    """ Checks if the last message contains tool calls """
    results=state['messages'][-1]
    return hasattr(results, "tool_calls") and len(results.tool_calls)>0

system_prompt="""
    You are an intelligent AI assistant who answers questions about Homelessness problems in Sf/California based on the PDF document loaded into the knowledge base.
    Use the retriever tool available to answer questions about homelessness problems in Sf/California. You can make multiple calls if needed.
    If you need to look up some information before asking a follow up question, you are allowed to do that.
    Please always cite the specific parts of the document you use in your answers.
"""

tool_dict={our_tool.name: our_tool for our_tool in tool}

# LLM Agent
def call_llm(state:AgentState)->AgentState:
    """ Function to call the LLM with the current state """
    # messages=list(state["messages"])
    messages=[SystemMessage(content=system_prompt)] + list(state["messages"])
    message=llm.invoke(messages)
    return {'messages':[message]}

# Retriever Agent
def take_action(state:AgentState)->AgentState:
    """ Function to execute tools from the LLM's response """
    tool_calls=state["messages"][-1].tool_calls
    results=[]
    for t in tool_calls:
        print(f"Executing tool: {t['name']} ... ")
        if not t['name'] in tool_dict:
            results.append(ToolMessage(
                tool_call_id=t['id'],
                name=t['name'],
                content=f"Tool {t['name']} not found!"
            ))
        else:
            result=tool_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Tool result: {result}")
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    print("Tools Executed!")
    return {'messages':results}

graph=StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_edge(START, 'llm')
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: 'retriever_agent',
        False: END
    }
)
graph.add_edge('retriever_agent', 'llm')

rag_agent=graph.compile()

def running_agent():
    while True:
        user_input=input("User question: ")
        if user_input.lower()=='exit':
            print("Exiting Agent.")
            break
        messages=[HumanMessage(content=user_input)]
        result=rag_agent.invoke({"messages": messages})
        print("Agent Response:")
        print(result['messages'][-1].content)


running_agent()
