from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

model1 = ChatGroq(model = "llama-3.3-70b-versatile")

model2 = ChatGroq(model = 'llama-3.1-8b-instant')

prompt1 = PromptTemplate(
    template = 'Generate short and simple notes from the following text \n {text}',
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = 'Generate 5 short question answers from the following text \n {text}',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz ->{quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Retrieval-Augmented Generation (RAG) is an AI architecture that combines information retrieval with large language models. 
Instead of relying only on pre-trained knowledge, a RAG system retrieves relevant information from external sources such as 
documents, PDFs, databases, or vector stores before generating a response. This improves factual accuracy and allows AI systems 
to answer questions about private or updated data.

A typical RAG pipeline includes document loading, text chunking, embedding generation, vector database storage, retrieval, and 
response generation. Embeddings convert text into numerical vectors that capture semantic meaning. Vector databases such as FAISS, 
Pinecone, and ChromaDB are used to efficiently search similar vectors.

When a user asks a question, the system converts the query into embeddings and retrieves the most relevant chunks from the vector 
database. These retrieved chunks are then passed to the language model as context. This helps the model generate more accurate 
and context-aware responses.

RAG is widely used in chatbots, customer support systems, enterprise search, document analysis, and AI assistants. LangChain is 
a popular framework for building RAG applications because it provides tools for document loading, prompt management, chaining, 
memory, and integrations with multiple vector databases and LLM providers.
"""
res = chain.invoke({'text':text})

print(res)