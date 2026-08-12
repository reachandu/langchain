from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    number_1: int
    number_2: int
    operation: str
    result: int

def add_node(state: AgentState) -> AgentState:
    """ Adds two numbers """
    state['result'] = state['number_1'] + state['number_2']
    return state

def subtract_node(state: AgentState) -> AgentState:
    """ Subtracts numbers """
    state['result'] = state['number_1'] - state['number_2']
    return state

def decide_next_node(state: AgentState) -> str:
    """ This node selects the next node of the graph based on a condition """
    if state['operation']=='+':
        return 'add_op'
    elif state['operation']=='-':
        return 'subtract_op'

graph=StateGraph(AgentState)

graph.add_node('add_node', add_node)
graph.add_node('subtract_node', subtract_node)
graph.add_node('router', lambda state:state)

graph.add_edge(START, 'router')
graph.add_conditional_edges(
    'router',
    decide_next_node,
    {
        'add_op': 'add_node',
        'subtract_op': 'subtract_node'
    }
)
graph.add_edge('add_node', END)
graph.add_edge('subtract_node', END)

bot=graph.compile()

input_state={
    'number_1': 15,
    'number_2': 5,
    'operation': '-'
}
result=bot.invoke(input_state)
print(result)