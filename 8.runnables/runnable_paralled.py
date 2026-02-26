from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda , RunnableParallel ,RunnableSequence # Needed for conversion step

# Load environment variables
load_dotenv()

# Initialize Gemini model and parser
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
parser = StrOutputParser()


prompt1 = PromptTemplate(
    template = "Generate a tweet on {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template = "Generate a linkdin post about  {topic}",
    input_variables= ['topic']
)


parallel_Chain = RunnableParallel({
            'tweet' : RunnableSequence(prompt1 , model , parser),
            'linkdin' : RunnableSequence(prompt2 , model , parser)
})

result = parallel_Chain.invoke({'topic' : 'AI'})
print(result['tweet'])
print(result['linkdin'])

print(result)