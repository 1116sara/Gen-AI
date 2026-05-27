'''Day 7 Assignment:
Build Day-7/review_query.py with these requirements:

Same laptop reviews you've been using since Day 4
First run → build index, save to disk
Every run after → load from disk, skip building
User types queries in a loop, quit to exit
For each query — print the answer AND print which source chunks were used with their scores
If you look at response.source_nodes — each node has .text and .score'''


from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

load_dotenv()

Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.chunk_size = 256
Settings.chunk_overlap = 20

texts = [
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
    print("Loading exisiting index from the disk.")
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)
    print("Index Loaded.")
else:
    print("No index found. Building from scratch...")
    print("Embedding documents....")
    documents = [Document(text=t) for t in texts]
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=index_path)

query_engine = index.as_query_engine(similarity_top_k=3) # Same as search() in faiss. Built-in function used to search the index.

while True:
    query = input("Enter your query: ")
    print()

    if query.strip().lower() == "quit":
        break

    response = query_engine.query(query)
    print(response.response)
    for node in response.source_nodes:
        print(f'Text: {node.text}')
        print(f'Score: {node.score:.4f}')
