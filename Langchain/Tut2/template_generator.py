from langchain_core.prompts import PromptTemplate


# template
template = PromptTemplate(
    template="""You are an expert AI researcher and academic writer.

Your task is to summarize a research paper based on the given requirements.

Research Paper: {paper}
Summary Style: {style}
Summary Length: {length}

Instructions based on style:
- If Beginner-Friendly: Use simple language, avoid jargon, use analogies and real-world examples
- If Technical: Use precise terminology, focus on methodology, architecture and implementation details
- If Code Oriented: Focus on practical implementation, include pseudocode or code snippets where relevant
- If Mathematical: Focus on equations, proofs, mathematical formulations and theoretical foundations

Instructions based on length:
- If Short (1-2 sentences): Capture only the core contribution and key result
- If Medium (1-2 paragraphs): Cover problem statement, approach and results
- If Long (1-2 pages): Cover background, problem, methodology, results, limitations and future work

Now write a {style} summary of the paper "{paper}" in {length} length.
""",
    input_variables=["paper", "style", "length"]
)

template.save("template.json")