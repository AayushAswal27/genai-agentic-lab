from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector = embedding.embed_query("Delhi is the capital of India")

print(len(vector))   # 768
print(vector[:5])