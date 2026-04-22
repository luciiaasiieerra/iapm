import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from generator import FinalExamGenerator

gen: FinalExamGenerator | None = None
gen_lock = threading.Lock()
gen_loading = False


def _load_generator_background():
    """Carga spaCy en un hilo separado para no bloquear el arranque."""
    global gen, gen_loading
    try:
        print("Cargando spaCy en background...")
        instance = FinalExamGenerator()
        with gen_lock:
            gen = instance
        print("Generador listo.")
    except Exception as e:
        print(f"Error cargando generador: {e}")
    finally:
        gen_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gen_loading
    gen_loading = True
    # Arrancar carga en hilo de fondo — el servidor ya está escuchando
    thread = threading.Thread(target=_load_generator_background, daemon=True)
    thread.start()
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


@app.get("/")
def root():
    # Render hace HEAD/GET aquí para detectar el puerto — responde SIEMPRE
    return {"status": "ok", "ready": gen is not None}


@app.get("/health")
def health():
    return {"status": "ok", "ready": gen is not None, "loading": gen_loading}


@app.post("/generate")
def generate(req: RequestModel):
    with gen_lock:
        current_gen = gen

    if current_gen is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo todavía cargando, reintenta en unos segundos."
        )

    result = current_gen.generate(req.text, existing_questions=req.existing_questions)
    result["preguntas"] = result["preguntas"][: req.max_questions]
    result["total"] = len(result["preguntas"])
    return result