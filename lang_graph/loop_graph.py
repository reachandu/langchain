from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, START, END
import random

class AgentState(TypedDict):
    name: str
    numbers: List[int]
    counter: int

def greeting_node(state:AgentState) -> AgentState:
    state['name']=f"Hi there, {state['name']}. lets loop through the nodes ..."
    state['counter'] = 0
    return state

def random_state(state:AgentState) -> AgentState:
    state['numbers'].append(random.randint(0, 10))
    state['counter']+=1
    return state

def should_continue(state:AgentState) -> str:
    if state['counter'] < 5:
        print(f"Enter the loop, {state['counter']}")
        return 'loop'
    else:
        print(f"Exiting the loop, {state['counter']}")
        return 'exit'

graph=StateGraph(AgentState)
graph.add_node('greet', greeting_node)
graph.add_node('random', random_state)

graph.add_edge(START, 'greet')
graph.add_edge('greet', 'random')

graph.add_conditional_edges(
    'random',
    should_continue,
    {
        'loop': 'random',
        'exit': END
    }
)

graph.add_edge('random', END)

bot=graph.compile()

input={
    'name': "Rob",
    'numbers': []
}

result=bot.invoke(input)
print(result)

