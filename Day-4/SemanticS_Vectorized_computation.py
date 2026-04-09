from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

#Simulated document store - 10 product reviews
documents = [
    "The battery life on this laptop is incredible, lasts all day",
    "Screen resolution is disappointing, colors look washed out",
    "Keyboard feels premium and typing experience is excellent",
    "The laptop runs very hot during heavy tasks, fan is loud",
    "Lightweight and portable, perfect for travel and commuting",
    "RAM is soldered and cannot be upgraded, big disappointment",
    "Boot time is blazing fast, SSD performance is outstanding",
    "Webcam quality is terrible, grainy even in good lighting",
    "Build quality feels solid, no flex in the chassis at all",
    "Price is too high for the specs you get with this machine"
]

# Step-1 : Embed all documents ONCE and store them.
#In a real system this happens once, then vectors are saved to disk.

print("Embedding documents...")
document_vectors = model.encode(documents)
print(f"Document matrix shape: {document_vectors.shape}")
print()

#Step-2 : Semantic search function.

def semantic_search(query, top_k=3, threshold=0.3):
    query_vector = model.encode(query)
    
    # Vectorized — computes ALL similarities in one operation
    # dot product of query against entire document matrix at once
    dot_products = np.dot(document_vectors, query_vector)
    
    # Normalize
    query_norm = np.linalg.norm(query_vector)
    doc_norms = np.linalg.norm(document_vectors, axis=1)
    similarities = dot_products / (doc_norms * query_norm)
    
    # Get top K indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Build results
    results = [
        (similarities[i], i, documents[i]) 
        for i in top_indices 
        if similarities[i] >= threshold
    ]
    return results

queries = [
    "How long does the battery last?",
    "Is the display good quality?",
    "How heavy is it to carry around?"
]

for query in queries:
    print(f"Query: '{query}'")
    print("-" * 50)
    results = semantic_search(query, top_k=2)
    for score, idx, doc in results:
        print(f"  Score: {score:.4f} | {doc}")
    print()