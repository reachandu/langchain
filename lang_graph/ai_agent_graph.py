from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-3.6-flash")

class AgentState(TypedDict):
    messages: List[str]

def process(state:AgentState) -> AgentState:
    response=model.invoke(state["messages"])
    state["messages"].append(response)
    print("AI: ", response.content[0].get('text'))
    return state

graph=StateGraph(AgentState)
graph.add_node('process', process)
graph.add_edge(START, 'process')
graph.add_edge('process', END)

bot=graph.compile()

user_input=input("Enter your message: ")
while user_input.lower()!='exit':
    input_state = {
        "messages": [HumanMessage(content=user_input)]
    }
    bot.invoke(input_state)
    user_input=input("Enter your message: ")

