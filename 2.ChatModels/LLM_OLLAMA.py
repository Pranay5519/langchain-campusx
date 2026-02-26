from langchain_ollama import Ollama

# Initialize LLM
llm = Ollama(
    model="llama3.2:latest",
    temperature=0
)

# Single prompt
response = llm.invoke("Explain Neural Networks in 3 lines.")

print(response)