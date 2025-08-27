from pydantic import BaseModel, Field


class Setup(BaseModel):
    project: str
    label_schema: str = Field(alias='schema')
    hostname: str
    access_token: str


class Task:
    def __init__(self, params):
        self.id = params['id']
        self.data = params['data']