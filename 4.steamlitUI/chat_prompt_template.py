from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_core.prompts import ChatPromptTemplate , load_prompt

chat_template = ChatPromptTemplate([
    ('system' , 'You are are helpfull {domain} Expert'),
    ('human' , 'Explain in simple terms what {topic}')
])

prompt = chat_template.invoke({'domain' : 'cricket','topic' : 'swing'})
print(prompt)