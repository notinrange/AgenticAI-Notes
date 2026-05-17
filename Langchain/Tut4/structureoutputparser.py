from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


load_dotenv()


model = ChatGroq(model="llama-3.1-8b-instant")

schema = [
    ResponseSchema(name="name", description="Name of the person"),
    ResponseSchema(name="age", description="Age of the person"),    
    ResponseSchema(name="city", description="City where the person lives")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me the name, age and city of a {topic} person \n {format_instructions}",
    input_variables=['topic'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.format({'topic':'fictional'})
# result = model.invoke(prompt)
# final_res = parser.parse(result.content)

chain = template | model | parser
final_res = chain.invoke({'topic':'fictional'})
print(final_res)