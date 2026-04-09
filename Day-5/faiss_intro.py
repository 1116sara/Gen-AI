from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

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
script_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(script_dir, "reviews.index")

if os.path.exists(index_path):
    print("Loading existing index from disk...")
    loaded_index = faiss.read_index(index_path)
    print(f"Index reloaded. Vectors in index: {loaded_index.ntotal}")

else:
    print("No index found. Building from scratch...")
    # Step - 1 Embed documents
    print("Embedding documents...")
    document_vectors = model.encode(documents)
    print(f"Matrix shape: {document_vectors.shape}")

    # Step-2 FAISS requires float32 specifically
    document_vectors = document_vectors.astype(np.float32)

    # Step-3 Create FAISS index
    # dimension = 384 (must watch your embedding model)
    dimension = document_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)

    print(f"Index created. Vectors in index: {index.ntotal}")

    # Step-4 Add vectors to index
    index.add(document_vectors)
    print(f"Vectors added. Vectors in index: {index.ntotal}")

    # Step-5 Save index to disk
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # index_path = os.path.join(script_dir, "reviews.index")

    faiss.write_index(index, index_path)
    print(f"Index saved. Vectors stored: {index.ntotal}")

# # Step-6 Reload index from disk
# loaded_index = faiss.read_index("reviews.index")
# print(f"Index reloaded. Vectors in index: {loaded_index.ntotal}")


# Adding more script

def search(query, k=3, distance_threshold=1.2):
    # Embed the query
    query_vector = model.encode([query])
    query_vector = query_vector.astype(np.float32)

    # FAISS search returns two arrays:
    # distances → L2 distances to nearest neighbors
    # indices   → positions of nearest neighbors in your documents list
    distances, indices = loaded_index.search(query_vector,k)

    print(f"\nQuery: '{query}'")
    print("-"*50)

    results = [
        (dist, idx) for dist, idx in zip(distances[0], indices[0])
        if dist <= distance_threshold
    ]

    if not results:
        print("No relevant results found.")
    else:
        for i, (dist,idx) in enumerate(results):
            print(f"  Rank {i+1} |  Distance: {dist:.4f} | {documents[idx]}")

search("How long does the battery last?")
search("Is the screen good?")
search("Does it come with free charger?")