from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from generator import FinalExamGenerator

app = FastAPI(
    title="Generador de Exámenes v3",
    description="Preguntas de opción múltiple sin APIs externas: patrones sintácticos + TF-IDF + comprensión global.",
    version="3.0.0",
)

gen = FinalExamGenerator()


class RequestModel(BaseModel):
    text: str = Field(..., min_length=50, description="Texto fuente")
    max_questions: int = Field(default=15, ge=1, le=40)


@app.post("/generate")
def generate(req: RequestModel):
    result = gen.generate(req.text)
    result["preguntas"] = result["preguntas"][: req.max_questions]
    result["total"] = len(result["preguntas"])
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
