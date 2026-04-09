from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Three sentences - two about rivers, one about finance
sentence_a = "The bank by the river was flooded"
sentence_b = "The river overflowed its banks after heavy rain"
sentence_c = "The bank refused my loan application"

# Get embeddings for all three
vector_a = model.encode(sentence_a)
vector_b = model.encode(sentence_b)
vector_c = model.encode(sentence_c)

# Cosine similarity function using numpy
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1,vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / magnitude

# Compare
sim_ab = cosine_similarity(vector_a, vector_b)
sim_ac = cosine_similarity(vector_a, vector_c)

print(f"Similarity A vs B (both about rivers): {sim_ab:.4f}")
print(f"Similarity A vs C (river vs finance): {sim_ac:.4f}")
print()

if sim_ab > sim_ac:
    print("✓ Model correctly understands context -- river sentences are closer than river vs finance")