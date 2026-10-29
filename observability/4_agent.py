import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_PROJECT"] = "agent"

import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'
    response = requests.get(url)
    return response.json()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, max_retries=5)

agent = create_react_agent(llm, tools=[search_tool, get_weather_data])

# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

response = agent.invoke({"messages": [("human", "What is the current temp of gurgaon")]})
print(response)
print(response["messages"][-1].content)