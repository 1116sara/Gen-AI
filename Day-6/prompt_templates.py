from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant",
               temperature=0.7,
               max_tokens=500)
# Multi Variable template
review_prompt = ChatPromptTemplate.from_messages([("system", """You are a {role}.
                                                  Your response must be in JSON format with these exact fields:
                                                  - sentiment: Positive, Negative, or Neutral
                                                  - confidence: float between 0 and 1
                                                  - summary: one line"""),
                                                  ("human", "Review: {review}")
                                                  ])

chain = review_prompt | llm

result = chain.invoke({
    "role": "sentiment analysis engine",
    "review" : "The product broke after 2 days but customer support was excellent."
})

print(result.content)
