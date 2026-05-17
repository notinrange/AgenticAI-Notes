from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.groq import GroqMessage

from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
messages = [
    SystemMessage(content="You are a helpful assistant that provides concise answers."),
    HumanMessage(content="What is the capital of France?"),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)