from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# This list IS the chatbot's memory
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("Chatbot ready. Type 'quit' to exit.\n")

while True:
    # Get user input
    user_input = input("You: ")
    
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
        messages=[conversation_history[0]] + conversation_history[-6:], #last 6 messages summarization.
        # for entire chat history use messages=conversation_history.
        temperature=0.7,
        max_tokens=500
    )
    
    # Extract assistant reply
    assistant_reply = response.choices[0].message.content
    
    # Append assistant reply to history
    conversation_history.append({
        "role": "assistant",
        "content": assistant_reply
    })
    
    print(f"\nBot: {assistant_reply}\n")
    print(f"\nTokens used in this call: {response.usage.completion_tokens}")
    print(f"\nTotal conversation tokens so far: {response.usage.total_tokens}")