from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a tweet about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a LinkedIn post about {topic}',
    input_variables = ['topic']
)

model = ChatGroq(model = "llama-3.3-70b-versatile")

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model, parser),
    'linkedIn' : RunnableSequence(prompt2, model, parser)
})

res = parallel_chain.invoke({'topic':'AI'})

print(res)