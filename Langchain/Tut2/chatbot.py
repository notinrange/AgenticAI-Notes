from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

chat_history = [
    SystemMessage(content="You are a helpful assistant that provides concise answers."),
]
while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("Bot:", result.content)


print("Chat history:", chat_history)