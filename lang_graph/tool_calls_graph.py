from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-3.6-flash")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int) -> int:
    """ add two numbers """
    return a+b

@tool
def sub(a:int, b:int) -> int:
    """ subtract two numbers """
    return a-b

@tool
def mul(a:int, b:int) -> int:
    """ multiply two numbers """
    return a*b

@tool
def div(a:int, b:int) -> int:
    """ devide two numbers """
    return a/b

my_tools=[add, sub, mul, div]
model=model.bind_tools(my_tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt=SystemMessage(content="You are an AI assistant, please answer my query to the best of your ability ")
    inputs=[system_prompt]+state['messages']
    response=model.invoke(inputs)
    return {"messages":state["messages"]+[response]}

def should_continue(state:AgentState):
    message=state["messages"]
    last_message=message[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph=StateGraph(AgentState)
graph.add_node("model_call", model_call)

tool_node=ToolNode(tools=my_tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "model_call")
graph.add_conditional_edges(
    'model_call',
    should_continue,
    {
        'end': END,
        'continue': "tools"
    }
)

graph.add_edge("tools", "model_call")
graph.add_edge("model_call", END)

bot=graph.compile()

def print_stream(stream):
    for s in stream:
        node_name, node_value=next(iter(s.items()))
        if 'messages' in node_value:
            message=['messages'][-1]
            if (isinstance(message, tuple)):
                print(message)
            else:
                message.pretty_print()

input={"messages": [("user", "add 4 and 2 and then calculate the product with 6. also tell me a joke at the end.")]}        
print_stream(bot.stream(input, stream_mode="values"))



