from dotenv import load_dotenv
from groq import Groq
import os

#load .env file into environment variables
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#SDK automatically reads GOOGLE_API_KEY from environment

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages =[
        # {"role": "system", "content": "You are an expert Python teacher who explains everything using simple analogies."},
        {"role":"user","content":"What is a token in LLMs?"}

    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)