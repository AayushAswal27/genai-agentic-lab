from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash",temperature=1.5, max_output_tokens=5)

result = model.invoke("Suggest me a poem")

print(result.text)