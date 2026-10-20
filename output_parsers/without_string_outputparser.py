from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto",
)

model = ChatHuggingFace(llm=llm) 

from langchain_core.prompts import PromptTemplate

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
) 

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Summarize the following text in 5 clear bullet points. Keep the summary simple and concise:\n{text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic": "black hole"})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result.content})

result1 = model.invoke(prompt2)

print(result1.content)