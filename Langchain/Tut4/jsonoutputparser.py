from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

# llm = HuggingFaceEndpoint(
#     model="google/gemma-2-2b-it",
#     task="text-generation",
#     max_new_tokens=512,
# )

# model = ChatHuggingFace(llm=llm)

model = ChatGroq(model="llama-3.1-8b-instant")

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# partial variable runitime se pehle hi set karna hota hai, kyunki ye prompt ke andar use ho raha hai

# prompt = template.format()

# print(prompt)

# result = model.invoke(prompt)
# final_res = parser.parse(result.content)

chain = template | model | parser
final_res = chain.invoke({})
print(final_res)