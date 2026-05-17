from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

st.header("Research Tool")

length_token_map = {
    "Short (1-2 sentences)": 100,
    "Medium (1-2 paragraphs)": 500,
    "Long (1-2 pages)": 2048
}
 
paper_input = st.selectbox("Select Research Paper", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners","Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox("Select Summary Style", ["Beginner-Friendly", "Technical", "Code Oriented", "Mathematical"])
length_input = st.selectbox("Select Summary Length", ["Short (1-2 sentences)", "Medium (1-2 paragraphs)", "Long (1-2 pages)"])

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7,max_tokens=length_token_map[length_input])

template = load_prompt("template.json")
# prompt = template.invoke({"paper": paper_input, "style": style_input, "length": length_input})

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({"paper": paper_input, "style": style_input, "length": length_input})
    # result = model.invoke(prompt)
    st.write(result.content)
