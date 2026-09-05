from pydantic import BaseModel

class Student(BaseModel):
    name: str = 'Aayush Aswal'

new_student = {}
student = Student(**new_student)
print(student) 