from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel , Field
from typing import Optional , Literal

# Load API key from .env
load_dotenv()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Define Structured Output with Pydantic Model
'''class Review(BaseModel):
    summary: str
    sentiment: Literal['pos','neg'] =Field(description="Return the sentiment of the review")
    pros : Optional[list[str]]=  Field(default = None , description= "Write Down all the Advantages of the product")
    cons : Optional[list[str]]=  Field(default = None , description= "Write Down all the Disadvantages of the product")
'''

json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}
# Enable structured output
s_model = llm.with_structured_output(json_schema)

# Product Review Text
review_text = '''
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, 
or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast 
charging is a lifesaver. The S-Pen integration is a great touch for note-taking and quick sketches, 
though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, 
capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant 
objects, but anything beyond 30x loses quality. However, the weight and size make it a bit uncomfortable 
for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different 
Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
'''

# Invoke Model
result = s_model.invoke(review_text)

# Print Result
print(result)
print(type(result))
