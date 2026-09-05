from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")


class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str,"Return sentiment of the review either negative, positive or neutral"
    ]


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
The hardware is excellent and feels really well-built.
The performance is smooth, and overall I'm quite happy with the experience.
The software has a lot of useful features, although I'd love to see a cleaner interface and
fewer pre-installed apps in future updates.
Hopefully, upcoming software updates will make the UI even more polished and modern.
Overall, a great device with plenty of potential!
""")

print(result)