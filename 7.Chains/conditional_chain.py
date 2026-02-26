from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# Pydantic Schema
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give the sentiment of the Feedback")

# Models and Parsers
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt Template with format instructions
prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative:\n"
             "{feedback}\n\n{format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

# Chain (using PydanticOutputParser)
classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)
chain = classifier_chain | branch_chain 

result = chain.invoke({'feedback':'This is terrible phone'})
print(result)
chain.get_graph().print_ascii()