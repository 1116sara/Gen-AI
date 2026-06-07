from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
from groq import Groq
import faiss
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
    # 2. Split into tokens (words are fine for now)
    # 3. Slide a window of chunk_size across the tokens
    #    advancing by (chunk_size - chunk_overlap) each step
    # 4. Join each window back into a string
    # 5. Return list of chunk strings

# print_chunks = load_and_chunk('sample_doc.txt')
# print(len(print_chunks))
# print(print_chunks[0])
# print()
# print(print_chunks[1])

script_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(script_dir,"review.index")

def build_index(chunks):
    doc_vectors = model.encode(chunks)
    doc_vectors = doc_vectors.astype(np.float32)
    dimension = doc_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    print(f"Index created.Vectors in index: {index.ntotal}")
    index.add(doc_vectors)
    print(f"Vectors added.Vectors in index: {index.ntotal}")
    return(index,chunks)
 # 1. Embed all chunks using sentence-transformers
    # 2. Convert to float32
    # 3. Create FAISS IndexFlatL2
    # 4. Add vectors to index
    # 5. Return both the index and the chunks list
# out_chunks = load_and_chunk('sample_doc.txt')
# index, chunks = build_index(out_chunks)

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
    # 1. Embed the query
    # 2. Convert to float32
    # 3. Search the index
    # 4. Return the top_k matching chunks as a list of strings

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
index, chunks = build_index(chunks)
# retrieved = retrieve("what are the symptoms of PCOD?", index, chunks)
# answer = generate("what are the symptoms of PCOD?", retrieved)
# print(answer)
retrieved = retrieve("what is the treatment cost of PCOD in India?", index, chunks)
answer = generate("what is the treatment cost of PCOD in India?", retrieved)
print(answer)
    # 1. Join retrieved chunks into a single context string
    # 2. Build the prompt with system message + context + question
    # 3. Call Groq API
    # 4. Return the answer