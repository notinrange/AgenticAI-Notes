from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7,max_tokens=20)

result = model.invoke("should i start with langchain or directly jump to lang graph ?")

print(result)