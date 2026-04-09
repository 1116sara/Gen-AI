from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is a token in LLMs? Answer in 2 sentences."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.usage.prompt_tokens)
print("----------------------")
print(response.usage.completion_tokens)
print("---------------------------------")
print(response.usage.total_tokens)

# print(type(response))

# # Extracting individual fields from the response object
# print("Response ID:", response.id)
# print("Model used:", response.model)
# print("Finish reason:", response.choices[0].finish_reason)
# print("Assistant reply:", response.choices[0].message.content)
# print("Role:", response.choices[0].message.role)
# print("Prompt tokens:", response.usage.prompt_tokens)
# print("Completion tokens:", response.usage.completion_tokens)
# print("Total tokens:", response.usage.total_tokens)


# response = client.chat.completions.create(
#     model="llama-3.1-8b-instant",
#     messages=[
#         {"role": "user", "content": "Give me a one sentence fun fact about space."}
#     ],
#     temperature=1.5,
#     max_tokens=100
#     # ask for 3 completions
# )

# # Now loop through all choices
# for i, choice in enumerate(response.choices):
#     print(f"Choice {i+1}: {choice.message.content}")
#     print("---")