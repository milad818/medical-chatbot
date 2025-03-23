

# create a prompt template
prompt_template = """
Answer the question below given the context. Please don't try to make up one if you don't know the answer.

Question: {question}
Context: {context}

Only return the helpful answer below and nothing else.
Helpful answer:
"""