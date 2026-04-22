from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from generator import FinalExamGenerator

gen: FinalExamGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gen
    gen = FinalExamGenerator()
    yield


app = FastAPI(
    title="Generador de Exámenes v3",
    description="Preguntas de opción múltiple sin APIs externas.",
    version="3.0.0",
    lifespan=lifespan,
)


class RequestModel(BaseModel):
    text: str = Field(..., min_length=50, description="Texto fuente")
    max_questions: int = Field(default=15, ge=1, le=40)
    existing_questions: list[str] = Field(
        default=[],
        description="Textos de preguntas ya guardadas en el banco. La IA no las repetirá."
    )


@app.get("/health")
def health():
    return {"status": "ok", "ready": gen is not None}


@app.post("/generate")
def generate(req: RequestModel):
    if gen is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo cargando, reintenta en unos segundos."
        )
    result = gen.generate(req.text, existing_questions=req.existing_questions)
    result["preguntas"] = result["preguntas"][: req.max_questions]
    result["total"] = len(result["preguntas"])
    return result