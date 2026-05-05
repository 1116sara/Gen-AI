from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import os

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant",
               temperature=0,
                 # for classification tasks always use temperature=0.,
               max_tokens=500)

review_prompt = ChatPromptTemplate.from_messages([("system", """You are a {role}.
                                                  Your response must be in JSON format with these exact fields:
                                                  - sentiment: Positive, Negative, or Neutral
                                                  - confidence: float between 0 and 1
                                                  - summary: one line"""),("human", "Review: {review}")
                                                  ])
# StrOutputParser — extracts just the string content from AIMessage
# Instead of result.content, the chain returns a plain string directly

string_chain = review_prompt | llm | StrOutputParser()

result = string_chain.invoke({
    "role": "Sentient analysis engine",
    "review": "Battery life is incredible but the screen is too dim."
})

print("Type:", type(result))
print("Result:", result)
print()

# JsonOutputParser — parses JSON string directly into Python dict

json_chain = review_prompt | llm | JsonOutputParser()

result = json_chain.invoke({
    "role": "sentiment analysis engine",
    "review": "Battery life is incredible but the screen is too dim."})

print("Type:", type(result))
print("Sentiment: ", result["sentiment"])
print("Confidence: ", result["confidence"])
print("Summary: ",result["summary"])