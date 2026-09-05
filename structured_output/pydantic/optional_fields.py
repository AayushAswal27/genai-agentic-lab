from pydantic import BaseModel
from typing import Optional

class Student(BaseModel):
    name: str = 'Aayush Aswal'
    age: Optional[int] = None

new_student = {}
student = Student(**new_student)
print(student)   