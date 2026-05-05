from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os

load_dotenv()

# This is LangChain's wrapper around the Groq API
# Remember Day 1 — you did this manually with client.chat.completions.create()
# LangChain is doing exactly that underneath

llm = ChatGroq(model="llama-3.1-8b-instant",
               temperature=0.7,
               max_tokens=500)

# .invoke() sends the message and returns a response
response = llm.invoke([SystemMessage(content="You are a helpful AI assistant."),
                       HumanMessage(content="What is a vector database in one sentence?")
                       ])

print("Response: ", response.content)
print("Type: ", type(response))
print("Full object: ", response)