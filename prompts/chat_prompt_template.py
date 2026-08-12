from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

chat_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("You are a helpful assistant that provide information about {topic}."),
        HumanMessagePromptTemplate.from_template("Can you tell me something interesting about {topic}?")
    ]
)
prompt = chat_prompt.format_prompt(topic="Langchain")
print(prompt)
print("---------------------------------------------------")
print(prompt.to_messages())
print("---------------------------------------------------")
print(prompt.to_string())