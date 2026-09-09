from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel  


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto",
)

model = ChatHuggingFace(llm=llm)

prompt1= PromptTemplate(
    template="Generate a tweet about a {topic}",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template="Generate a Linkedin post about a {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

parallel_chain= RunnableParallel({
    "tweet" : RunnableSequence(prompt1,model,parser),
    "linkedin" : RunnableSequence(prompt2,model,parser)}
) 
result=parallel_chain.invoke({"topic":"GPT6-Astra"})
print(result)