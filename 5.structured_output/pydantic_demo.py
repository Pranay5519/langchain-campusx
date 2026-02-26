from pydantic import   BaseModel , EmailStr , Field
from typing import Optional


class Student(BaseModel):
    name : str = 'pranay'
    age : Optional[int] =None
    cgpa : float =Field(gt=0 , lt=10 ,description='Decimal value representing CGPA')
new_Student = {'age'  : '32' , 'cgpa' : 4.55} 
student = Student(**new_Student)

print(student.age , student.name)
student_json = student.model_dump_json()
print(student_json)