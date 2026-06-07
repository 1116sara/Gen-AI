# LangChain Personal Reference Sheet
> My quick-lookup document. Only patterns I've actually used and understood.
> Last updated: Day 6

---

## When to use LangChain
- I need full control over the prompt structure
- I'm building chains, agents, or custom pipelines
- I need to connect LLMs with tools, memory, or APIs
- I want to customize exactly what gets injected into the prompt

## When to use LlamaIndex instead
- I have documents that need intelligent chunking and retrieval
- I want a full RAG pipeline without wiring everything manually

---

## Install

```bash
uv pip install langchain langchain-groq langchain-huggingface langchain-community
```

---

## 1. Imports

```python
from langchain_groq import ChatGroq                          # LLM
from langchain_core.prompts import ChatPromptTemplate        # prompt builder
from langchain_core.output_parsers import StrOutputParser    # parse LLM output to string
from langchain_core.messages import HumanMessage, AIMessage  # message types
from langchain_core.chat_history import InMemoryChatMessageHistory  # memory store
from langchain_core.runnables.history import RunnableWithMessageHistory  # memory wrapper
```

---

## 2. The LLM

```python
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)
```

---

## 3. LCEL — The Pipe Operator

```python
# Chain components together with |
chain = prompt | llm | parser

# Each component receives output of previous component
# prompt output → llm input
# llm output    → parser input
# parser output → your result
```

**This replaced the old LangChain v1 `.run()` syntax.**
**Every component in a chain is a Runnable — it has .invoke(), .stream(), .batch()**

---

## 4. ChatPromptTemplate — building prompts

```python
# Basic prompt with variables
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# With chat history for memory
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{chat_history}"),  # history gets injected here
    ("human", "{input}")
])
```

**Variables in curly braces get filled in at runtime.**
**("placeholder", "{chat_history}") expands the full message list into the prompt.**

---

## 5. Output Parsers

```python
# StrOutputParser — converts LLM response object to plain string
parser = StrOutputParser()

# Without parser — you get a full AIMessage object
# With parser    — you get just the text string
```

---

## 6. Running a Chain

```python
chain = prompt | llm | StrOutputParser()

# .invoke() — single call, returns result
response = chain.invoke({"input": "your question here"})
print(response)  # plain string

# .stream() — streams tokens as they generate
for chunk in chain.stream({"input": "your question"}):
    print(chunk, end="", flush=True)
```

---

## 7. Memory — Conversation History

```python
# Step 1 — Create a store for all sessions
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Step 2 — Wrap chain with memory
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Step 3 — Always pass session_id when invoking
config = {"configurable": {"session_id": "user_123"}}
response = chain_with_memory.invoke(
    {"input": "Hi, my name is Sara"},
    config=config
)
```

**Why session_id:** Multiple users can have separate conversation histories.
Same chain, different session_id = different memory.

---

## 8. How Memory Works Internally

```
Turn 1: history = []
        prompt = system + [] + "Hi my name is Sara"
        → LLM responds → history now has 1 human + 1 AI message

Turn 2: history = [HumanMessage("Hi..."), AIMessage("Hello Sara...")]
        prompt = system + [history] + "What did I tell you my name was?"
        → LLM sees full history → knows the answer is Sara
```

**LangChain handles the appending automatically when using RunnableWithMessageHistory.**
**The model itself has no memory — memory lives in your store dictionary.**

---

## 9. Message Types

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

HumanMessage(content="user said this")
AIMessage(content="assistant said this")
SystemMessage(content="system instruction")

# Check history
history = get_session_history("user_123")
for message in history.messages:
    print(type(message).__name__, message.content)
```

---

## 10. Day 6 Mapping — LangChain vs Raw API

| Day 6 LangChain | Day 2 Raw API equivalent |
|----------------|--------------------------|
| `ChatGroq(model=...)` | `Groq(api_key=...)` |
| `ChatPromptTemplate` | manually building messages list |
| `StrOutputParser` | `response.choices[0].message.content` |
| `chain = prompt \| llm \| parser` | separate function calls in sequence |
| `RunnableWithMessageHistory` | your manual conversation_history list |
| `store = {}` | your conversation_history list |
| `session_id` | no equivalent — raw API had one session only |

---

## 11. Common Mistakes

```
✗ Using old LangChain v1 syntax like LLMChain, .run()
  → Use LCEL pipe operator instead

✗ Forgetting config={"configurable": {"session_id": "..."}} when invoking
  → Memory won't work without it

✗ Using ("placeholder", "{chat_history}") without the placeholder keyword
  → History won't expand into individual messages

✗ Expecting the LLM to remember across sessions automatically
  → Memory is in your store dict, not the model. Model is always stateless.

✗ Parsing response when using RunnableWithMessageHistory
  → The wrapper handles output — don't double-parse
```

---

## 12. Full Working Template — Chain with Memory

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

config = {"configurable": {"session_id": "user_1"}}

print("Chatbot ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        break
    response = chain_with_memory.invoke(
        {"input": user_input},
        config=config
    )
    print(f"Bot: {response}\n")
```
