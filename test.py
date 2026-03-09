from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-120b",
    max_tokens=4000
)

response = llm.invoke("what are transformers")

print(response)
print("===="*40)
print(response.content)