from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    number_1: int
    number_2: int
    number_3: int
    number_4: int
    operation_1: str
    operation_2: str
    result_1: int
    result_2: int

def add_node_1(state: AgentState) -> AgentState:
    """ Adds two numbers """
    state['result_1'] = state['number_1'] + state['number_2']
    return state

def subtract_node_1(state: AgentState) -> AgentState:
    """ Subtracts numbers """
    state['result_1'] = state['number_1'] - state['number_2']
    return state

def add_node_2(state: AgentState) -> AgentState:
    """ Adds two numbers """
    state['result_2'] = state['number_3'] + state['number_4']
    return state

def subtract_node_2(state: AgentState) -> AgentState:
    """ Subtracts numbers """
    state['result_2'] = state['number_3'] - state['number_4']
    return state

def decide_next_node_1(state: AgentState) -> str:
    """ This node selects the next node of the graph based on a condition """
    if state['operation_1']=='+':
        return 'add_op_1'
    elif state['operation_1']=='-':
        return 'subtract_op_1'

def decide_next_node_2(state: AgentState) -> str:
    """ This node selects the next node of the graph based on a condition """
    if state['operation_2']=='+':
        return 'add_op_2'
    elif state['operation_2']=='-':
        return 'subtract_op_2'
    
graph=StateGraph(AgentState)

graph.add_node('add_node_1', add_node_1)
graph.add_node('subtract_node_1', subtract_node_1)
graph.add_node('router_1', lambda state:state)

graph.add_node('add_node_2', add_node_2)
graph.add_node('subtract_node_2', subtract_node_2)
graph.add_node('router_2', lambda state:state)

graph.add_edge(START, 'router_1')
graph.add_conditional_edges(
    'router_1',
    decide_next_node_1,
    {
        'add_op_1': 'add_node_1',
        'subtract_op_1': 'subtract_node_1'
    }
)

graph.add_edge('add_node_1', 'router_2')
graph.add_edge('subtract_node_1', 'router_2')
graph.add_conditional_edges(
    'router_2',
    decide_next_node_2,
    {
        'add_op_2': 'add_node_2',
        'subtract_op_2': 'subtract_node_2'
    }
)

graph.add_edge('add_node_2', END)
graph.add_edge('subtract_node_2', END)

bot=graph.compile()

input_state={
    'number_1': 15,
    'number_2': 5,
    'number_3': 10,
    'number_4': 20,
    'operation_1': '-',
    'operation_2': '+'    
}
result=bot.invoke(input_state)
print(result)