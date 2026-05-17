from langchain_text_splitters import RecursiveCharacterTextSplitter
text = """
Income Tax in India is governed by the Income Tax Act, 1961. 
Every individual whose total income exceeds the basic exemption 
limit is required to file an Income Tax Return (ITR).

GST (Goods and Services Tax) was introduced in India on 
1st July 2017. It replaced multiple indirect taxes like 
VAT, service tax, and excise duty. GST has four tax slabs: 
0%, 5%, 12%, 18%, and 28%.

A Chartered Accountant (CA) is responsible for auditing 
financial statements, tax planning, and ensuring compliance 
with tax laws. CAs play a crucial role in GST filing, 
ITR preparation, and financial advisory services.

TDS (Tax Deducted at Source) is a mechanism where tax is 
deducted at the point of income generation itself. Employers 
deduct TDS from employee salaries before disbursement.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0,
)

chunks = splitter.split_text(text)

print(len(chunks))