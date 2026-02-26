from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Pydantic Model
class Football(BaseModel):
    what: str = Field(description="Tell me about Football")
    how: str = Field(description="How Football is played")
    goat: str = Field(description="Greatest Player of all time in Football")

# Parser
parser = PydanticOutputParser(pydantic_object=Football)

# Prompt with format instructions from parser
template = PromptTemplate(
    template="Write a short 5 lines Essay on {topic}. \n{format_instructions}",
    input_variables=['topic'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Chain: Prompt → Model → Parser
chain = template | model | parser

# Run the Chain
final_output = chain.invoke({'topic': "Football"})

print(final_output)
