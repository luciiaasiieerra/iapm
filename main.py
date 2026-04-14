from fastapi import FastAPI
from pydantic import BaseModel
from generator import FinalExamGenerator

app = FastAPI()
gen = FinalExamGenerator()

class RequestModel(BaseModel):
    text: str

@app.post("/generate")
def generate(req: RequestModel):
    return gen.generate(req.text)