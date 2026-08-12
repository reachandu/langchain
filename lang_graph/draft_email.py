from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# global variable
document_content=""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content:str)-> str:
    """ updates the document with provided content """
    global document_content
    document_content=content
    return f"Document has been updated successfully. The document content is :\n{document_content}"

@tool
def save(file_name:str) -> str:
    """
        Save the current document_content to a given file
        Args:
            file_name: name of a text file 
    """
    global document_content
    if not file_name.endswith('.txt'):
        file_name+='.txt'

    try:
        with open(file_name, "w") as file:
            file.write(document_content)
        print(f"Document content saved successfully as {file_name}")
        return f"Document content saved successfully as {file_name}"
    except Exception as e:
        print(f"Error saving document: {e}")
        return f"Error saving document: {e}"

tools=[update, save]
model=model.bind_tools(tools)

my_tools=ToolNode(tools)

def our_agent(state:AgentState)-> AgentState:
    system_prompt=SystemMessage(content=f"""
        You are an email drafter: a helpful writing assistant. You are going to help the user update and modify documents.
        - If the user wnats to update or modify content, use 'update' tool with the complete updated content
        - If the user wants to save and finish, you need to use the tool save.
        - make sure to always show the current document content.

        The current document contetnt is : {document_content}
    """)
    if not state['messages']:
        user_input=input("Im ready to help you update a document. What would you like to create?")
        user_message=HumanMessage(content=user_input)
    else:
        user_input=input("What would you like to do with this document?")
        print(f"user: {user_input}")
        user_message=HumanMessage(content=user_input)
    all_messages=[system_prompt]+ list(state['messages'])+[user_message]

    response=model.invoke(all_messages)
    print(f"AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"Invoking tool: {[tc['name'] for tc in response.tool_calls]}\n")
    return {'messages':list(state['messages'])+[user_message, response]}

def should_continue(state:AgentState) -> str:
    """ Determine if we should continue or end the conversation """
    messages=state['messages']
    if not messages:
        return "continue"
    for message in reversed(messages):
        if(isinstance(message, ToolMessage)) and 'saved' in message.content.lower() and 'document' in message.content.lower():
            return "end"
    return "continue"

graph=StateGraph(AgentState)
graph.add_node('agent', our_agent)
graph.add_node('tools', my_tools)

graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

bot=graph.compile()

def print_messages(messages):
    """ To print the messages in more readable format """
    if not messages:
        return
    for message in messages[-3]:
        if isinstance(message, ToolMessage):
            print(f"ToolMessage: {messages.content}")

def run_document_agent():
    state={'messages':[]} 
    for step in bot.stream(state):
        if 'messages' in step:
            print_messages[step['messages']]
    print("Conversation ended")


run_document_agent()