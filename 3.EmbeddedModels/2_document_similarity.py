from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Virat Kohli is one of the finest batsmen in the world. He has led the Indian cricket team in all three formats and is known for his aggressive style and consistency across formats.",
    
    "MS Dhoni, the former Indian captain, is celebrated for his calm demeanor and incredible finishing skills. Under his leadership, India won the 2007 T20 World Cup, 2011 ODI World Cup, and the 2013 Champions Trophy.",
    
    "Sachin Tendulkar, also known as the 'Master Blaster', holds numerous records in international cricket including 100 international centuries and over 34,000 international runs.",
    
    "Rohit Sharma is known for his elegance and ability to score big hundreds. He is the only player to have scored three double centuries in One Day Internationals.",
    
    "Jasprit Bumrah is India's leading fast bowler, known for his unorthodox action, deadly yorkers, and ability to bowl in the death overs. He has been crucial in both Tests and limited overs cricket."
]

test_query = "Greatest player of all time?"

doc_embedding = embedding.embed_documents(documents)
query_embedding  = embedding.embed_query(test_query)

scores =  cosine_similarity([query_embedding] , doc_embedding)[0]
print(scores)
print(sorted(list(enumerate(scores)),key=lambda x:x[1])[-1])