# Hugging face sentence- transformers.
from sentence_transformers import SentenceTransformer

# This downloads the model locally on first run (~90MB)
# After that it's cached — no internet needed
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get embedding for a single sentence
vector = model.encode("The bank by the river was flooded")

print("Type:", type(vector))
print("Dimensions:", len(vector))
print("First 5 values:", vector[:5])




# # # Groq embedding not working

#  from dotenv import load_dotenv
#  from groq import Groq
#  import os
#  load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#  #Get embedding for a single sentence
# response = client.embeddings.create(
#      model="nomic-embed-text-v1.5",
#      input="The bank by the river was flooded."
#  )

# vector = response.data[0].embedding

# print("Type:",type(vector))
# print("Dimensions:",len(vector))
# print("First 5 values:", vector[:5])


