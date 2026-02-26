from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize Chat Model
chat = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)

# Send messages
response = chat.invoke([
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain what is Machine Learning in 3 lines.")
])

print(response.content)
