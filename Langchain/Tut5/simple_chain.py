from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")

prompt = PromptTemplate(
    template = 'Generate 5 interesting facts about {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

chain.get_graph().print_ascii()

res = chain.invoke({'topic':'AI Engineering'})



print(res)

