from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "HuggingFaceH4/zephyr-7b-beta",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm, temperature=0, max_tokens=20)

result = model.invoke("should i start with langchain or directly jump to lang graph ?")
print(result.content)