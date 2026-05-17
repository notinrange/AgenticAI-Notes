from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

result = llm.invoke("What is the capital of India?")

print(result.content)

# this is not llm it is chat model api, so we need to use chat model api to get the response from the model.