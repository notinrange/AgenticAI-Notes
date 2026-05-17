from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")
prompt1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template = 'Summarize the following text \n {topic}',
    input_variables=['text']
)
parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1,model,parser)
branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

res = final_chain.invoke({'topic':'Russia vs Ukraine'})

print(res)