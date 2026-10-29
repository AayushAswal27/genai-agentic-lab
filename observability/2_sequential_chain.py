import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'sequential-llm-app'

model1 = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite', temperature=0.7, max_retries=5)
model2 = ChatGoogleGenerativeAI(model='gemini-3.8-flash', temperature=0.5, max_retries=5)

prompt1 = PromptTemplate.from_template('Generate a detailed report on {topic}')
prompt2 = PromptTemplate.from_template('Generate a 5 point summary from the following text:\n{text}')
parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'tags': ['llm_app', 'report_generation', 'summarization'],
    'metadata': {
        'model1': 'gemini-3.1-flash-lite',
        'model1_temp': 0.7,
        'model2': 'gemini-3.8-flash',
        'model2_temp': 0.5,
        'parser': 'StrOutputParser',
    }, 
    'run_name': 'sequential_chain',
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)
print(result) 