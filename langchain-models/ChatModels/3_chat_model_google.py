from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash",temperature=0)

result = model.invoke("Suggest me 5 Indian male names")

print(result.text)