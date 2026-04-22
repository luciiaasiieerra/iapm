from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from generator import FinalExamGenerator

gen: FinalExamGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # NO cargamos spaCy aquí — Render mataría el proceso por timeout
    # La carga se hace lazy en el primer /generate
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


def get_generator() -> FinalExamGenerator:
    """Carga el generador la primera vez que se necesita (lazy loading)."""
    global gen
    if gen is None:
        print("Cargando spaCy por primera vez...")
        gen = FinalExamGenerator()
        print("Generador listo.")
    return gen


@app.get("/")
def root():
    # Render hace HEAD / para detectar el puerto — debe responder inmediatamente
    # sin esperar a que spaCy cargue
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "ready": gen is not None}


@app.post("/generate")
def generate(req: RequestModel):
    generator = get_generator()  # carga spaCy aquí si aún no está cargado
    result = generator.generate(req.text, existing_questions=req.existing_questions)
    result["preguntas"] = result["preguntas"][: req.max_questions]
    result["total"] = len(result["preguntas"])
    return result