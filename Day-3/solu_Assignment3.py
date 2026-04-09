from dotenv import load_dotenv
from groq import Groq
import os
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# This list IS the chatbot's memory
conversation_history = [
    {"role": "system", 
     "content": """You are a sentiment analysis engine. Return only valid raw JSON with no markdown, no backticks, no explanation. Nothing except the JSON object with exactly these four fields:
        sentiment: string, must be exactly one of these three values :Positive, Negative,or Neutral. no other values allowed.
        reasoning: string (one single sentence explaning the classification)
        confidence: float between 0 and 1
        summary: string (one line summary of the review)"""}
]

print("Sentiment Analysis Chatbot ready. Type 'quit' to exit.\n")

while True:
    # Get user input
    user_input = input("Enter Your Product review: ")
    
    if user_input.lower() == "quit":
        break
    
    # Append user message to history
    conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Send entire history to API
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=conversation_history, #last 6 messages summarization.
        # for entire chat history use messages=conversation_history.
        temperature=0,
        max_tokens=200
    )
    
    # Extract assistant reply
    assistant_reply = response.choices[0].message.content
    
    # Append assistant reply to history
    conversation_history.append({
        "role": "assistant",
        "content": assistant_reply
    })
    
    print(f"\nBot: {assistant_reply}\n")


    try:
        parsed = json.loads(assistant_reply)
        print("Sentiment:", parsed["sentiment"])
        print("Reasoning::", parsed["reasoning"])
        print("Confidence:", parsed["confidence"])
        print("Summary:", parsed["summary"])

        if parsed["confidence"] < 0.85:
            print("Flag for human review.")
        else:
            print("Passed to dashboard.")
    except json.JSONDecodeError:
        print("Model returned malformed JSON. Raw output: ", assistant_reply)