from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I've 
been using it for two weeks now. The Snapdragon 8 Gen 3 processor makes everything lightning fast. 
The 5000mAh battery easily lasts a full day. The 45W fast charging is a lifesaver.

However, the weight and size make it a bit uncomfortable for one-handed use. Samsung's One UI still 
comes with bloatware, and the $1,300 price tag is quite steep.

Pros:
- Insanely powerful processor
- Stunning 5000mAh battery life
- Blazing-fast 45W charging

Review by Aayush Aswal
""")

print(result)
