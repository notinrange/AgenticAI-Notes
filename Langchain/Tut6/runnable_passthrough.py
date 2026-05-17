from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()



prompt1 = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template = 'Explain the following {text}',
    input_variables = ['text']
)


model = ChatGroq(model = "llama-3.3-70b-versatile")

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'AI Engineering'}))