from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is one of the greatest batsmen in cricket history.",
    "Rohit Sharma is the current captain of the Indian cricket team.",
    "MS Dhoni is famous for his legendary finishing and calm captaincy.",
    "Jasprit Bumrah is the best fast bowler India has ever produced.",
    "Hardik Pandya is a powerful all-rounder who can bat and bowl."
]

# embed all documents
doc_embeddings = embedding.embed_documents(documents)

# embed a query
query = "Who is the best batsman?"
query_embedding = embedding.embed_query(query)

# find most similar document

# in cosine similarity space input are in 2D, so we need to reshape the query embedding to be 2D
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
most_similar_index = np.argmax(similarities)

print(f"Query: {query}")
print(f"Most similar: {documents[most_similar_index]}")
print(f"Similarity score: {similarities[most_similar_index]:.4f}")