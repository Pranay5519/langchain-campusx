from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_core.prompts import PromptTemplate , load_prompt
import re
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task  = "text-generation"
)

model = ChatHuggingFace(llm = llm)
prompt = load_prompt(r'steamlitUI\chatbot_template.json')

chat_history  = [
    SystemMessage(content ='you are a helpfull Human Assistant')
]
while True :
    user_input = input("you : ")
    chat_history.append(HumanMessage(content = user_input))
    if user_input =='exit':
        break
    result = model.invoke(f'{chat_history} , {prompt}')
    clean_output = re.sub(r'<think>.*?</think>', '', result.content, flags=re.DOTALL).strip()
    chat_history.append(AIMessage(content= clean_output))
    print("AI : " ,clean_output)
#langchain has built in function that does all the above steps..
print(chat_history)

