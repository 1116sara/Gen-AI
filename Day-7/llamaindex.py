from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
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
]

documents = [Document(text=t) for t in texts]

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)

response = query_engine.query("How long does the battery last?")
print(response.response)