from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")

prompt = PromptTemplate(
    template = 'Answer the following question \n {question} from the document - \n {text}',
    input_variables = ['question','text']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/apple-macbook-pro-m5-16-gb-1-tb-ssd-macos-tahoe-mde14hn-a/p/itm14a21c70f80c5'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

res = chain.invoke({'question':'What is the Brand of this product?','text':docs[0].page_content})
print(res)