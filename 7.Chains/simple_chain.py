#from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


prompt = PromptTemplate(
    template = "generate 5 interesing facts about {topic}",
    input_variable  = ['topic']
)

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

chain = prompt | model  | parser 

result = chain.invoke({'topic' : 'cricket'})

print(result)

chain.get_graph().print_ascii()
