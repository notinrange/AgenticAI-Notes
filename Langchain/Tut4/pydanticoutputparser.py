from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me the name, age and city of a fictional {place} person \n {format_instructions}",
    input_variables=['place'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.format(place='Indian')
# result = model.invoke(prompt) 
# final_res = parser.parse(result.content)

chain = template | model | parser
final_res = chain.invoke({'place':'Indian'})
print(final_res)