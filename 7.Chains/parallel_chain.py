from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()


model1 = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
model2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

prompt1 = PromptTemplate(
    template = "Generate short and simple note from the follwing text\n {'text}",
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = "generate 5 Question and Answers from the from the Follwing text\n{text}",
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and questions into a single Document \n notes--{notes} , quiz -->{quiz}',
    input_variables = ['notes','quiz']
)

parser = StrOutputParser()
parallel_chain =RunnableParallel({
    'notes' : prompt1  | model1 | parser ,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain 

text = """
    A self-driving car, also known as an autonomous car (AC), driverless car, robotic car or robo-car,[1][2][3] is a car that is capable of operating with reduced or no human input.[4][5] They are sometimes called robotaxis, though this term refers specifically to self-driving cars operated for a ridesharing company. Self-driving cars are responsible for all driving activities, such as perceiving the environment, monitoring important systems, and controlling the vehicle, which includes navigating from origin to destination.[6]
As of late 2024, no system has achieved full autonomy (SAE Level 5). In December 2020, Waymo was the first to offer rides in self-driving taxis to the public in limited geographic areas (SAE Level 4),[7][failed verification] and as of April 2024 offers services in Arizona (Phoenix) and California (San Francisco and Los Angeles). In June 2024, after a Waymo self-driving taxi crashed into a utility pole in Phoenix, Arizona, all 672 of its Jaguar I-Pace vehicles were recalled after they were found to have susceptibility to crashing into pole-like items and had their software updated.[8][9][10] In July 2021, DeepRoute.ai started offering self-driving taxi rides in Shenzhen, China. Starting in February 2022, Cruise offered self-driving taxi service in San Francisco,[11] but suspended service in 2023. In 2021, Honda was the first manufacturer to sell an SAE Level 3 car,[12][13][14] followed by Mercedes-Benz in 2023.[15]
"""

result = chain.invoke(chain)

print(result)
chain.get_graph().print_ascii()