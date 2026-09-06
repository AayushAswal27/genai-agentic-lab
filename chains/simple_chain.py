from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto",
)

model = ChatHuggingFace(llm=llm) 

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)
parser = StrOutputParser()

chain = prompt| model | parser
result = chain.invoke({"topic": "Cricket"})
print(result)

chain.get_graph().print_ascii(). #Visualizing the chain