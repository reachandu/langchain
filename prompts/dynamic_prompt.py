from langchain_core.prompts import PromptTemplate

dynamic_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Give a funfact about {topic} in a {style} style."
    )

prompt = dynamic_prompt.format_prompt(topic="AI", style="humorous")
print(prompt.to_string())