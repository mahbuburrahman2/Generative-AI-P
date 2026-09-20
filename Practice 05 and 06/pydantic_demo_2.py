from pydantic import BaseModel, EmailStr,Field
from typing import Optional

class JobApplication(BaseModel):
    name: str ="unknown"

    experience: Optional[int]=None

    email: EmailStr

    expected_salary : int = Field( gt =0 , description="Expected Annual Salary of the candidate" )


new_application ={
    "email": "abs@gmial.com",
    "experience": "3",
    "expected_salary": 500,
}

application = JobApplication(**new_application)
print(application)

#convert pydantic object to json

application_json = application.model_dump_json()

print(application_json)

