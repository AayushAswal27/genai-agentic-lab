from pydantic import BaseModel

class Student(BaseModel):
    name: str

new_student = {'name': 'Aayush Aswal'}
student = Student(**new_student)

print(student)
print(type(student))