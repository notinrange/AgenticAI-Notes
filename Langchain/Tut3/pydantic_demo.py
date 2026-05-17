from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name: str = 'rahul'
    age: Optional[int] = None
    email: EmailStr 
    cgpa : float = Field(gt=0.0, lt=10.0,default=5.0, description="CGPA must be between 0.0 and 10.0")

new_student = {'name': 'Rahul', 'age': '30','email':'abf@gmail.com', 'cgpa': '8.5'}

student = Student(**new_student)

# we can make pydantic model into dict or json like this

# student = dict(student)
student = student.model_dump_json() # this will give us json string

print(student)