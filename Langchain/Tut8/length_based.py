from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# text = """
# Income Tax in India is governed by the Income Tax Act, 1961. 
# Every individual whose total income exceeds the basic exemption 
# limit is required to file an Income Tax Return (ITR).

# GST (Goods and Services Tax) was introduced in India on 
# 1st July 2017. It replaced multiple indirect taxes like 
# VAT, service tax, and excise duty. GST has four tax slabs: 
# 0%, 5%, 12%, 18%, and 28%.

# A Chartered Accountant (CA) is responsible for auditing 
# financial statements, tax planning, and ensuring compliance 
# with tax laws. CAs play a crucial role in GST filing, 
# ITR preparation, and financial advisory services.

# TDS (Tax Deducted at Source) is a mechanism where tax is 
# deducted at the point of income generation itself. Employers 
# deduct TDS from employee salaries before disbursement.
# """

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator=''
)

# result = splitter.split_text(text  )
result = splitter.split_documents(docs)

print(result[0].page_content)