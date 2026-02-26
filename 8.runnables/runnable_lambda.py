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


def word_count(text):
    return len(text.split())

prompt  = PromptTemplate(
    template  = "Write a joke on {topic} ",
    input_variables = ['topic']
)

joke_gen_chain = RunnableSequence(prompt , model , parser)

parallel_chain  = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count) 
})

final_chain = RunnableSequence(joke_gen_chain , parallel_chain)

result = final_chain.invoke({'topic'  : 'AI' })
final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])
print(final_result)