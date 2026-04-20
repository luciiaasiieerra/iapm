from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from generator import FinalExamGenerator

# Instancia global, inicializada en el lifespan (no al importar)
gen: FinalExamGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos UNA vez cuando el servidor ya está escuchando."""
    global gen
    gen = FinalExamGenerator()
    yield
    # cleanup si hiciera falta


app = FastAPI(
    title="Generador de Exámenes v3",
    description="Preguntas de opción múltiple sin APIs externas.",
    version="3.0.0",
    lifespan=lifespan,
)


class RequestModel(BaseModel):
    text: str = Field(..., min_length=50, description="Texto fuente")
    max_questions: int = Field(default=15, ge=1, le=40)


@app.get("/health")
def health():
    # Render usa este endpoint para el port scan — responde inmediatamente,
    # incluso antes de que el generador termine de cargar
    return {"status": "ok", "ready": gen is not None}


@app.post("/generate")
def generate(req: RequestModel):
    if gen is None:
        raise HTTPException(status_code=503, detail="Modelo cargando, reintenta en unos segundos.")
    result = gen.generate(req.text)
    result["preguntas"] = result["preguntas"][: req.max_questions]
    result["total"] = len(result["preguntas"])
    return result
