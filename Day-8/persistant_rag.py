from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
from groq import Groq
import faiss
import json
import os

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_and_chunk(filepath, chunk_size=200, chunk_overlap=50):
    with open(filepath,'r',encoding="utf-8") as file:
        words = file.read().split()
    chunks = []
    for position in range(0,len(words),(chunk_size - chunk_overlap)):
        window = words[position:(chunk_size + position)]
        chunk = " ".join(window)
        chunks.append(chunk)
    return(chunks)

script_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(script_dir,"rag.index")
chunks_path = os.path.join(script_dir,"chunks.json")

def build_or_load_index(chunks):
    if os.path.exists("rag.index") and os.path.exists("chunks.json"):
        print("Loading existing index from disk...")
        index = faiss.read_index(index_path)
        with open("chunks.json", "r") as f:
            saved_chunks = json.load(f)
        print(f"Index reloaded: Vectors in index: {index.ntotal}")
        return(index,saved_chunks)

    else:
        print("No index found. Building from scratch...")
        print("Embedding documents....")
        doc_vectors = model.encode(chunks)
        doc_vectors = doc_vectors.astype(np.float32)
        dimension = doc_vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        print(f"Index created.Vectors in index: {index.ntotal}")
        index.add(doc_vectors)
        print(f"Vectors added.Vectors in index: {index.ntotal}")
        faiss.write_index(index, index_path)
        print(f"Index saved. Vectors stored: {index.ntotal}")
        with open("chunks.json", "w") as f:
            json.dump(chunks,f)
        return(index,chunks)

def retrieve(query, index, chunks, top_k=3):
    query_vector = model.encode([query])
    query_vector = query_vector.astype(np.float32)
    distances, indices = index.search(query_vector, top_k)
    results = [ (dist, idx) for dist, idx in zip(distances[0],indices[0])]
    matching_chunks = []
    for dist, idx in results:
        preparind_list = chunks[idx]
        matching_chunks.append(preparind_list)
    return matching_chunks

def generate(query, retrieved_chunks):
    string = " ".join(retrieved_chunks)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role" : "system",
                "content" : "Answer only using the provided context. If the answer is not present in the context, say I don't have enough information to answer this. Do not use outside knowledge."
            },
            {
                "role" : "user",
                "content" : f"Context: {string}\n\n Question: {query}"
            }
        ], 
        temperature=0,
        max_tokens=200
    )
    
    llm_reply = response.choices[0].message.content

    return llm_reply

chunks = load_and_chunk('sample_doc.txt')
index, chunks = build_or_load_index(chunks)
retrieved = retrieve("what is the treatment cost of PCOD in India?", index, chunks)
answer = generate("what is the treatment cost of PCOD in India?", retrieved)
print(answer)