from pathlib import Path
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto",
)
model = ChatHuggingFace(llm=llm)

loader = TextLoader("/Users/aayushaswal/genai-agentic-lab/document_loader/poem.txt", encoding="utf-8")
docs = loader.load()

prompt = PromptTemplate(
    template="Give me a short summary of the following:\n{text}",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"text": docs[0].page_content})
print(result)