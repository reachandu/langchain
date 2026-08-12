from typing import TypedDict, Dict
from langgraph.graph import StateGraph, START, END

class MsgState(TypedDict):
    message: str

def greeting_node(state: MsgState) -> MsgState:
    """
        Simple node that addes greeting message to the state.
    """
    state['message']="Hello " + state['message'] + ", Welcome to LangGraph learning!"
    return state

graph=StateGraph(MsgState)
graph.add_node('initial', greeting_node)
graph.add_edge(START, 'initial')
graph.add_edge('initial', END)

bot=graph.compile()

response=bot.invoke(MsgState(message='Chandu'))

print(response)