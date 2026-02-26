from langchain_core.prompts import PromptTemplate

# template
template = PromptTemplate(
    template="""Please act like  you are chatbot and answer the user question"""
)

template.save('chatbot_template.json')