from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.7,max_tokens=20)

result = model.invoke("should i start with langchain or directly jump to lang graph ?")  

print(result.content)