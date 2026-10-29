import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_PROJECT"] = "agent"

import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    key = os.getenv("WEATHERSTACK_KEY")
    url = f'https://api.weatherstack.com/current?access_key={key}&query={city}'
    response = requests.get(url)
    return response.json()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, max_retries=5)

agent = create_agent(llm, tools=[search_tool, get_weather_data])



response = agent.invoke({"messages": [("human", "What is the current temp of gurgaon")]})
print(response)
print(response["messages"][-1].content)