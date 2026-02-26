from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import re

load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Chat Prompt Template
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful {domain} expert, so answer only in 5 to 10 words, do not provide additional information, since you are a chatbot'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# Load chat history
chat_history = []
try:
    with open(r'steamlitUI\chat_history.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Human:"):
                chat_history.append(HumanMessage(content=line.replace("Human:", "").strip()))
            elif line.startswith("AI:"):
                chat_history.append(AIMessage(content=line.replace("AI:", "").strip()))
except FileNotFoundError:
    pass  # If file doesn't exist yet

# Chat Loop
while True:
    user_input = input("you: ")
    if user_input.lower() == 'exit':
        break

    # Prepare prompt
    prompt = chat_template.invoke({
        'chat_history': chat_history,
        'domain': 'customer support',
        'query': user_input
    })

    # Add human message to history
    chat_history.append(HumanMessage(content=user_input))

    # Get model response
    result = model.invoke(prompt)
    clean_output = re.sub(r'<think>.*?</think>', '', result.content, flags=re.DOTALL).strip()

    # Add AI message to history
    chat_history.append(AIMessage(content=clean_output))
    print("AI:", clean_output)

# Save chat history (optional)
with open(r'steamlitUI\chat_history.txt', 'w') as f:
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            f.write(f"Human: {msg.content}\n")
        elif isinstance(msg, AIMessage):
            f.write(f"AI: {msg.content}\n")



#python steamlitUI\message_placeholder.py