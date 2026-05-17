from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


load_dotenv()

llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm, temperature=0)
# model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

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

prompt1 = template1.invoke({'topic':'black holes'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result2 = model.invoke(prompt2)

print(result2.content)