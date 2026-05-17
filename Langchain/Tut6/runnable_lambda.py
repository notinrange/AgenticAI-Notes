from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda

load_dotenv()

def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)

prompt = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)

model = ChatGroq(model = "llama-3.3-70b-versatile")
parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : runnable_word_counter
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'AI Engineering'}))