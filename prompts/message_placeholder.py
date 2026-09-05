from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# create the chat template WITH a placeholder for history
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# load the past chat history  
chat_history = []
with open("/Users/aayushaswal/genai-agentic-lab/prompts/chat_history.txt") as f: 
    chat_history.extend(f.readlines())

print(chat_history)   

#create the final prompt by filling both slots
prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "Where is my refund"
})

print(prompt)