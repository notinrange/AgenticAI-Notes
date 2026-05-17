from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)

model = ChatGroq(model = "llama-3.3-70b-versatile")

parser = StrOutputParser()

chain = RunnableSequence(prompt, model,parser)

print(chain.invoke({'topic':'AI Engineeing'}))