from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant",
               temperature=0.7,
               max_tokens=500)
# Define a reusable prompt template
# {topic} is a variable that gets filled in at runtime

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI engineer. Be concise."),
    ("human","Explain {topic} in exactl 2 sentences.")
])

# Build the chain using pipe operator.
chain = prompt | llm

# Invoke the chain with different topics
topics = ["RAG","Vector database","cosine similarity"]

for topic in topics:
    result = chain.invoke({"topic": topic})
    print(f"\nTopic: {topic}")
    print(f"Answer: {result.content}")
