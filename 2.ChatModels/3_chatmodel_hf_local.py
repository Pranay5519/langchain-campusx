from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import os

os.environ["HF_HOME"] = "C:/Users/prana/Desktop/VS_CODE/LangChainModels"

pipe = pipeline(
   task  = "text-generation",
    model="deepseek-ai/DeepSeek-R1-0528",
    max_new_tokens=50 
)

llm = HuggingFacePipeline(pipeline=pipe)
result = llm.invoke("What is the capital of India?")
print(result)

#python 2.ChatModels\3_chatmodel_hf_local.py
