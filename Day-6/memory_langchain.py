from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Prompt with a placeholder for chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

# Session store — each session_id gets its own history
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap chain with message history
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Conversation — same session_id maintains history
config = {"configurable": {"session_id": "user_001"}}

print("Turn 1:")
r1 = chain_with_memory.invoke(
    {"input": "My name is Tharuni and I am learning GenAI."},
    config=config
)
print(r1.content)

print("\nTurn 2:")
r2 = chain_with_memory.invoke(
    {"input": "What am I learning?"},
    config=config
)
print(r2.content)

print("\nTurn 3:")
r3 = chain_with_memory.invoke(
    {"input": "What is my name?"},
    config=config
)
print(r3.content)