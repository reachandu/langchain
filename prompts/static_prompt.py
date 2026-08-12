from langchain_core.prompts import PromptTemplate

static_prompt = PromptTemplate(
    input_variables=[],
    template="Give me a funfact about Langchain."
    )

prompt = static_prompt.format_prompt()

print(prompt.to_string())