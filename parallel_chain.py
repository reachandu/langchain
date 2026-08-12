from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model_name = "llama-3.3-70b-versatile"
model1=ChatGroq(model=model_name)
model2=ChatGroq(model=model_name)

prompt_template_1 = PromptTemplate(
    input_variables=["text"],
    template="""
        Generate short and simple note for the text: {text}.
    """
)

prompt_template_2 = PromptTemplate(
    input_variables=["text"],
    template="""
        Generate five short questions and answers for the text: {text}.
    """
)

prompt_template_3 = PromptTemplate(
    input_variables=["notes", "qa"],
    template="""
        Merge the notes and questions and answers into a single text. Notes: {notes}. Questions and Answers: {qa}.
    """
)

parser = StrOutputParser()

runnable_chain = RunnableParallel(
    {
        'notes': prompt_template_1 | model1 | parser,
        'qa': prompt_template_2 | model2 | parser
    }
)   
merge_chain = prompt_template_3 | model1 | parser
final_chain = runnable_chain | merge_chain

about_utah = """
    Utah is located in the western United States and is known for its stunning natural landscapes.
    Salt Lake City is the capital and largest city of Utah.
    Utah is home to five famous national parks, often called the "Mighty Five."
    The Great Salt Lake is one of the largest saltwater lakes in the Western Hemisphere.
    Utah has world-class ski resorts that attract visitors from around the globe each winter.
    The state's economy is supported by technology, tourism, mining, and outdoor recreation.
    Utah features diverse scenery, including mountains, deserts, canyons, and forests.
    Many popular outdoor activities in Utah include hiking, mountain biking, rock climbing, and camping.
    Utah has a rich cultural history influenced by Native American tribes and early pioneer settlers.
    Utah is consistently recognized as one of the best states for outdoor adventure and quality of life.
"""

result = final_chain.invoke({"text": about_utah})
print("---------------------------------------------------")
print("Model:", model_name)
print("Response:", result)
