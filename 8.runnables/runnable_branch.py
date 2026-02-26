from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableLambda, RunnableBranch, RunnablePassthrough, RunnableParallel, RunnableSequence
)

def word_count(text):
    return len(text.split())

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text:\n{text}',
    input_variables=['text']
)

report_chain = prompt1 | model | parser

# ✅ Here, we apply branching on the report (string), not on dict
conditions = RunnableBranch(
    (lambda text: len(text.split()) > 50, prompt2 | model | parser),
    RunnablePassthrough()
)


parallel_chain = RunnableParallel({
    'length': report_chain | RunnableLambda(word_count),
    'summary_or_not': report_chain | conditions
})

result = parallel_chain.invoke({'topic': 'Agentic AI'})
print(result)
