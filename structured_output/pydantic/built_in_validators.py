from pydantic import BaseModel, EmailStr
from typing import Optional

class Student(BaseModel):
    name: str = 'Aayush Aswal'
    age: Optional[int] = None
    email: EmailStr

new_student = {'age': '19', 'email': 'abc'}    # invalid email
student = Student(**new_student)   # ERROR: value is not a valid email address

new_student = {'age': '32', 'email': 'abc@gmail.com'}   # valid
student = Student(**new_student)   # works
print(student)