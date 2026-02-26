from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda  # Needed for conversion step

# Load environment variables
load_dotenv()

# Initialize Gemini model and parser
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
parser = StrOutputParser()

# Prompt to write a joke
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# Prompt to explain a joke
prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# Define a step to wrap the joke string into {"text": ...} for prompt2
format_for_prompt2 = RunnableLambda(lambda x: {"text": x})

# Build the chain using pipes
chain = prompt1 | model | parser | format_for_prompt2 | prompt2 | model | parser

# Run the chain
result = chain.invoke({'topic': 'AI'})
print(result)
