from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
import json

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a sentiment analysis engine. Return only valid raw JSON with no markdown, no backticks, no explanation. Nothing except the JSON object."},
        {"role": "user", "content": "Classify this sentence as positive, negative or neutral. Return exactly these three fields: sentiment, reasoning, confidence (0 to 1). Sentence: 'The product arrived late but worked perfectly.'"}
    ],
    temperature=0,
    max_tokens=200
)

assistant_reply = response.choices[0].message.content
print("Raw output:", assistant_reply)

# Parse the JSON string into a Python dictionary
parsed = json.loads(assistant_reply)
print("Sentiment:", parsed["sentiment"])
print("Confidence:", parsed["confidence"])
