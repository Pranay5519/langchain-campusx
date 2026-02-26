from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate , load_prompt

load_dotenv()

# Set up LLM
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content= 'You are a Human Assistant'),
    HumanMessage(content = 'Tell me about langchain')
]

result = model.invoke(messages)
messages.append(AIMessage(content = result.content ))
print(messages)