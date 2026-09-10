from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display(self):
        print(f"{self.name} is {self.age} years old")

s = Student("Aayush Aswal", 19 )
if s.age > 18:
    print("adult")
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])