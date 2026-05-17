from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage


# chat template with placeholder
chat_template = ChatPromptTemplate([
    ('system', "You are a helpful customer support assistant that provides concise answers."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', "{query}"),
])

#load chat history from file
chat_history = []
#loading previous chat history
try:
    with open("chat_history.txt") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("Human:"):
                chat_history.append(HumanMessage(content=line.replace("Human:", "").strip()))
            elif line.startswith("Assistant:"):
                chat_history.append(AIMessage(content=line.replace("Assistant:", "").strip()))
except FileNotFoundError:
    print("No chat history found. Starting fresh.")

print(chat_history)

#create prompt with chat history and new query

prompt = chat_template.invoke({"chat_history": chat_history, "query": "What is the return policy?"})
print(prompt)
