from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda , RunnableParallel ,RunnableSequence ,RunnablePassthrough # Needed for conversion step

# Load environment variables
load_dotenv()

# Initialize Gemini model and parser
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
parser = StrOutputParser()

passthrough  =RunnablePassthrough()

prompt1 = PromptTemplate(
    template = "Write a joke on this {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template = "Expain the following joke :   {topic}",
    input_variables= ['topic']
)

joke_gen_chain = RunnableSequence(prompt1 , model , parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explaination'  : RunnableSequence(prompt2 , model , parser)
})

final_chain =  RunnableSequence(joke_gen_chain , parallel_chain)

result=  final_chain.invoke("AI")
print(result)