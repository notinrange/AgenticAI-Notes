# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# llm = HuggingFaceEndpoint(
#     model="meta-llama/Llama-3.2-3B-Instruct",
#     task="text-generation",
#     max_new_tokens=512,
#     provider ="nebius"
# )

# model = ChatHuggingFace(llm=llm, temperature=0)
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# 1st prompt
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)


# 2nd prompt
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black holes'})

print(result)