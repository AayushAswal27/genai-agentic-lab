from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Aayush Aswal'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')

new_student = {'age': '19', 'email': 'abc@gmail.com', 'cgpa': 12}   # 12 > 10 → ERROR
student = Student(**new_student)

new_student = {'age': '19', 'email': 'abc@gmail.com', 'cgpa': 5}    # valid
student = Student(**new_student)
print(student)