from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    name: str
    age: int
    skills: List[str]
    result: str

def first_node(state: AgentState) -> AgentState:
    """        First node         """
    state['result'] = state['name'] + " welcome to the Fist Node!"
    return state

def second_node(state: AgentState) -> AgentState:
    """        Second node         """
    skills = ", ".join(state['skills'])
    state['result'] = f"{state['name']} you have these skills {skills}"
    return state

def third_node(state: AgentState) -> AgentState:
    """        Third node         """
    state['result'] = state['result'] + f" you are {state['age']} years old"
    return state

graph=StateGraph(AgentState)
graph.add_node('first', first_node)
graph.add_node('second', second_node)
graph.add_node('third', third_node)

graph.add_edge(START, 'first')
graph.add_edge('first', 'second')
graph.add_edge('second', 'third')
graph.add_edge('third', END)

bot=graph.compile()

response=bot.invoke(AgentState(name='Chandu', age=25, skills=['Python', 'Java', 'ML']))

print(response)