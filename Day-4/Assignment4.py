# Review_search.py
"""
Day 4 Assignment
Before Day 5, extend your semantic search to handle a real use case. Build Day-4/review_search.py with these requirements:
1. A list of at least 15 product reviews (mix of topics)
2. User types a search query in the terminal
3. System returns top 3 relevant reviews above threshold 0.3
4. Display the score alongside each result
5. If no results above threshold → print "No relevant reviews found"
6. Loop so user can search multiple times, 'quit' to exit
"""

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "This laptop is the best for editing videos and high performance tasks",
    "The display quality is premium but marks on screen are visible once started using.",
    "Battery life is remarkable. Lasts all day.",
    "Build quality looks luxury.",
    "Keyboard feels premium and typing experience is excellent",
    "Lightweight and portable, perfect for travel and commuting",
    "The laptop runs very hot during heavy tasks, cooling system works okay okay.",
    "Price is too high for the specs you get with this machine",
    "Auto screen off when you close the laptop is awesome.",
    "Very sleek design body. Doesnot need 2 hands to open.",
    "Not sure If I would upgrade my RAM because basic version only costs more. ",
    "Basic computing requirements, but runs smooth than other OS",
    "Only has two Thunderbolt 3 ports and 4 USB-C ports, No HDMI.",
    "Tasks which involve High-end graphics degrade as internal temperature rise and risks unexpected shutdowns.",
    "Cannot upgrade CPU and GPU. Need to buy new device."
]

print("......Embedding Documents......")
document_vectors = model.encode(documents)
print(f"Document matrix shape: {document_vectors.shape}")
print()

# semantic search function
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
    matches = [(similarities[i], i, documents[i]) for i in top_indices if similarities[i] >= threshold]
    return matches



while True:
    #Get user query
    query = input("Your query: ")
    print()

    if query.lower() == "quit":
        break
    
    print(f"Query: '{query}'")
    print("-" * 50)
    search_results = semantic_search(query, top_k=3,threshold=0.3)

    if not search_results:
         print("No relevant reviews found.")
    else:
        for score, idx, doc in search_results:
            print(f"  Score: {score:.4f}| {idx} | {doc}")
    print()

