# LlamaIndex Personal Reference Sheet
> My quick-lookup document. Only patterns I've actually used and understood.
> Last updated: Day 7

---

## When to use LlamaIndex
- I have documents (PDFs, text files, etc.) I need to make queryable
- I need intelligent chunking, not manual splitting
- I need document metadata and chunk relationships preserved
- I need a full RAG pipeline without wiring everything manually

## When to use LangChain instead
- I need full control over the prompt
- I'm building agents, custom chains, or non-document pipelines

## When to use both
- LlamaIndex for document ingestion and retrieval
- LangChain for orchestration and custom prompt control on top

---

## Install

```bash
uv pip install llama-index llama-index-llms-groq llama-index-embeddings-huggingface
```

---

## 1. Imports

```python
from llama_index.core import (
    VectorStoreIndex,        # index type for semantic search
    Document,                # container for your text + metadata
    Settings,                # global config for LLM and embedding model
    StorageContext,          # needed for saving/loading index
    load_index_from_storage  # loads saved index from disk
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
```

---

## 2. Settings — always set this first

```python
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.chunk_size = 256      # must match embedding model max input
Settings.chunk_overlap = 20    # tokens repeated between chunks — prevents broken sentences
```

**Why:** LlamaIndex does chunking, embedding, and synthesis automatically.
It needs to know which models to use before doing any of that.

---

## 3. Creating Documents

```python
# From a list of strings
documents = [Document(text=t) for t in texts]

# From a single string
doc = Document(text="some content")

# With metadata attached
doc = Document(
    text="some content",
    metadata={"filename": "policy.pdf", "page": 3}
)
```

**Why not just pass plain strings:** Document is a container.
It holds text + metadata + allows LlamaIndex to track relationships between chunks.
Plain strings have no metadata, no structure — LlamaIndex doesn't know what to do with them.

---

## 4. Building an Index

```python
# From documents — does chunk + embed + store in one call
index = VectorStoreIndex.from_documents(documents)
```

**What happens internally:**
1. NodeParser splits each Document into Nodes (chunks)
2. embed_model.encode(node.text) for each node
3. Stores vectors + text + metadata + relationships together

**This is Days 4 and 5 packaged into one line.**

---

## 5. Index Types — which one to use

| Index | Use when |
|-------|----------|
| `VectorStoreIndex` | Semantic search — most production RAG systems |
| `SummaryIndex` | Summarization over small fixed document sets — LLM reads everything |
| `TreeIndex` | Very long hierarchical documents |
| `KeywordTableIndex` | Exact keyword lookup needed |

**Default choice: VectorStoreIndex**

---

## 6. Persistence — Save and Load

```python
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(script_dir, "my_index")  # directory, not a file

# Pattern — same logic as Day 5 FAISS, different API
if os.path.exists(index_path):
    # Load from disk — instant, no re-embedding
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)
else:
    # Build from scratch — pay embedding cost once
    documents = [Document(text=t) for t in texts]
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=index_path)  # save everything
```

**Key difference from FAISS:**
- FAISS saves one `.index` binary file
- LlamaIndex saves a directory — vectors + node text + metadata + relationships

---

## 7. Query Engine — searching the index

```python
# Create once, outside the query loop
query_engine = index.as_query_engine(similarity_top_k=3)

# Query
response = query_engine.query("your question here")
```

**What happens internally:**
1. Embeds the query
2. Finds top K similar nodes (similarity_top_k controls how many)
3. Builds prompt: context chunks + question
4. Sends to LLM
5. Returns Response object

**This is your Day 5 search() function + Day 6 prompt injection, built-in.**

---

## 8. Reading the Response

```python
response = query_engine.query("How long does the battery last?")

# The synthesized answer — LLM read chunks and produced this
print(response.response)

# The raw chunks that were retrieved
for node in response.source_nodes:
    print(node.text)    # the chunk content
    print(node.score)   # similarity score — higher = more relevant
```

**source_nodes = what was RETRIEVED**
**response.response = what was SYNTHESIZED**

These are two different steps. Retrieval finds relevant chunks.
Synthesis is the LLM reading those chunks and producing a human answer.

---

## 9. Chat Engine — query engine with memory

```python
# For conversational use — remembers previous messages
chat_engine = index.as_chat_engine()
response = chat_engine.chat("How long does the battery last?")
response = chat_engine.chat("What about the keyboard?")  # remembers context
```

---

## 10. Common Mistakes

```
✗ Passing plain strings to from_documents()
  → Use Document objects: [Document(text=t) for t in texts]

✗ Setting Settings after building the index
  → Always set Settings before any index operations

✗ Using .ntotal to check vector count (that's FAISS)
  → LlamaIndex doesn't have .ntotal

✗ Saving to a file path instead of a directory
  → persist_dir expects a folder, not a file like "reviews.index"

✗ Creating query_engine inside the while loop
  → Create once outside the loop, reuse for every query
```

---

## 11. Full Working Template

```python
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

texts = ["your documents here"]

script_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(script_dir, "my_index")

if os.path.exists(index_path):
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)
else:
    documents = [Document(text=t) for t in texts]
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=index_path)

query_engine = index.as_query_engine(similarity_top_k=3)

while True:
    query = input("Query: ").strip()
    if query.lower() == "quit":
        break
    response = query_engine.query(query)
    print(f"Answer: {response.response}")
    for node in response.source_nodes:
        print(f"  {node.score:.4f} | {node.text}")
```
