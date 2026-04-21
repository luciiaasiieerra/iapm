"""
FinalExamGenerator v4
=====================
4 tipos de pregunta, máximo una por categoría:
  - PER   : ¿Quién? — persona detectada por spaCy NER
  - LOC   : ¿Dónde? — lugar detectado por spaCy NER
  - DATE  : ¿Cuándo? — año/período detectado por regex
  - DEF   : ¿Qué es? — solo si "X es Y" con sujeto y predicado cortos y claros

Distractores: siempre del mismo tipo, extraídos del texto.
"""

import re
import json
import random
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

import spacy


# ──────────────────────────────────────────────
# Normalización y stopwords
# ──────────────────────────────────────────────

def normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _load_stopwords() -> set:
    import spacy.lang.es
    result = set()
    for w in spacy.lang.es.STOP_WORDS:
        result.add(normalize(w))
    return result


STOPWORDS: set = _load_stopwords()


# ──────────────────────────────────────────────
# Regex
# ──────────────────────────────────────────────

RE_YEAR = re.compile(r"\b(?P<year>(?:19|20)\d{2})\b")
RE_PERIOD = re.compile(
    r"\b(?P<period>siglo\s+[IVXLC]+|d[eé]cada(?:\s+de(?:\s+los)?)?\s+\d{4}s?)\b",
    re.IGNORECASE,
)
RE_DEF = re.compile(
    r"^(?P<sujeto>[A-ZÁÉÍÓÚ][^.]{2,40}?)\s+"
    r"(?:es|son|fue|era|se define como|se denomina|se considera|se llama)\s+"
    r"(?P<predicado>[^.]{5,80})$",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────
# Candidatos por tipo desde el texto
# ──────────────────────────────────────────────

def _build_pool(doc) -> dict:
    """
    Extrae candidatos del propio texto agrupados por tipo.
    Solo fuentes: NER de spaCy + regex de fechas.
    """
    pool = defaultdict(set)
    full = doc.text

    # Fechas
    for m in RE_YEAR.finditer(full):
        pool["DATE"].add(m.group("year"))
    for m in RE_PERIOD.finditer(full):
        pool["DATE"].add(m.group("period"))

    # Entidades
    for ent in doc.ents:
        if ent.label_ == "PER":
            pool["PER"].add(ent.text.strip())
        elif ent.label_ in ("GPE", "LOC"):
            pool["LOC"].add(ent.text.strip())
        elif ent.label_ == "ORG":
            pool["ORG"].add(ent.text.strip())

    return {k: list(v) for k, v in pool.items()}


# ──────────────────────────────────────────────
# Distractores
# ──────────────────────────────────────────────

def _distractors(answer: str, label: str, pool: dict, n: int = 3) -> list:
    """
    Devuelve n distractores del MISMO tipo semántico que la respuesta.
    Para DATE genera años cercanos si faltan candidatos.
    Para el resto, si no hay suficientes en el texto, devuelve lista vacía
    → la pregunta se descarta.
    """
    answer_norm = normalize(answer)
    candidates = list(pool.get(label, []))

    # DATE: rellenar con años aritméticamente
    if label == "DATE":
        m = RE_YEAR.search(answer)
        if m:
            base = int(m.group())
            offsets = [o for o in range(-30, 31) if o != 0]
            random.shuffle(offsets)
            for o in offsets:
                y = str(base + o)
                if y not in candidates:
                    candidates.append(y)
                if len(candidates) >= n + 5:
                    break

    random.shuffle(candidates)
    result = []
    for c in candidates:
        cn = normalize(c)
        if cn == answer_norm:
            continue
        if SequenceMatcher(None, cn, answer_norm).ratio() < 0.6:
            result.append(c.strip())
        if len(result) == n:
            break

    # Segunda pasada con umbral más laxo
    if len(result) < n:
        for c in candidates:
            cn = normalize(c)
            if cn != answer_norm and c.strip() not in result:
                result.append(c.strip())
            if len(result) == n:
                break

    return result


# ──────────────────────────────────────────────
# Extracción de preguntas por tipo (todas las del texto)
# ──────────────────────────────────────────────

def _all_per_questions(doc, pool: dict) -> list:
    """Devuelve UNA pregunta por cada oración con entidad PER distinta."""
    seen = set()
    results = []
    other_pers = pool.get("PER", [])

    for sent in doc.sents:
        ents = [e for e in sent.ents if e.label_ == "PER"]
        if not ents:
            continue
        ent = ents[0]
        key = normalize(ent.text)
        if key in seen:
            continue
        seen.add(key)

        distractors = [p for p in other_pers if normalize(p) != key]
        if len(distractors) < 3:
            continue

        ctx = sent.text.replace(ent.text, "_____").strip()
        results.append({
            "tipo": "persona",
            "pregunta": f"¿Qué persona completa la siguiente afirmación?\n→ {ctx}",
            "respuesta": ent.text.strip(),
            "label": "PER",
        })
    return results


def _all_loc_questions(doc, pool: dict) -> list:
    """Devuelve UNA pregunta por cada oración con entidad LOC/GPE distinta."""
    seen = set()
    results = []
    other_locs = pool.get("LOC", [])

    for sent in doc.sents:
        ents = [e for e in sent.ents if e.label_ in ("GPE", "LOC")]
        if not ents:
            continue
        ent = ents[0]
        key = normalize(ent.text)
        if key in seen:
            continue
        seen.add(key)

        distractors = [l for l in other_locs if normalize(l) != key]
        if len(distractors) < 3:
            continue

        ctx = sent.text.replace(ent.text, "_____").strip()
        results.append({
            "tipo": "lugar",
            "pregunta": f"¿En qué lugar ocurre lo siguiente?\n→ {ctx}",
            "respuesta": ent.text.strip(),
            "label": "LOC",
        })
    return results


def _all_date_questions(doc, pool: dict) -> list:
    """Devuelve UNA pregunta por cada oración con fecha/año distinto."""
    seen = set()
    results = []

    for sent in doc.sents:
        m = RE_YEAR.search(sent.text)
        if not m:
            m = RE_PERIOD.search(sent.text)
            if not m:
                continue
            fecha = m.group("period")
        else:
            fecha = m.group("year")

        key = normalize(fecha)
        if key in seen:
            continue
        seen.add(key)

        ctx = sent.text[:m.start()] + "_____" + sent.text[m.end():]
        results.append({
            "tipo": "fecha",
            "pregunta": f"¿En qué año o período ocurre lo siguiente?\n→ {ctx.strip()}",
            "respuesta": fecha,
            "label": "DATE",
        })
    return results


def _all_def_questions(doc, pool: dict) -> list:
    """
    Devuelve preguntas de definición solo cuando hay alta confianza:
    - Sujeto ≤ 4 palabras con mayúscula
    - Predicado 8-60 chars, no empieza por artículo
    - Distractores: otros predicados de otras definiciones del mismo texto
    """
    bad_starts = {"un","una","el","la","los","las","que","cuando",
                  "donde","como","este","esta","algo","muy"}

    # Recoger todos los pares (sujeto, predicado) válidos primero
    pairs = []
    for sent in doc.sents:
        m = RE_DEF.match(sent.text.strip())
        if not m:
            continue
        sujeto = m.group("sujeto").strip()
        predicado = m.group("predicado").strip()
        if len(sujeto.split()) > 4:
            continue
        if len(predicado) < 8 or len(predicado) > 60:
            continue
        if normalize(predicado.split()[0]) in bad_starts:
            continue
        if not any(c.isupper() for c in sujeto):
            continue
        pairs.append((sujeto, predicado, sent.text.strip()))

    if len(pairs) < 4:
        return []  # Necesitamos ≥4 para tener 3 distractores por pregunta

    results = []
    for i, (sujeto, predicado, _) in enumerate(pairs):
        distractors = [p for j, (_, p, _) in enumerate(pairs) if j != i]
        random.shuffle(distractors)
        results.append({
            "tipo": "definicion",
            "pregunta": f"¿Cómo se define o describe «{sujeto}»?",
            "respuesta": predicado,
            "label": "DEF",
            "distractores_custom": distractors[:3],
        })
    return results


# ──────────────────────────────────────────────
# Generador principal
# ──────────────────────────────────────────────

class FinalExamGenerator:
    def __init__(self):
        print("Cargando spaCy...")
        try:
            self.nlp = spacy.load("es_core_news_sm")
        except OSError:
            import os
            os.system("python -m spacy download es_core_news_sm")
            self.nlp = spacy.load("es_core_news_sm")
        print("Sistema listo.")

    def _filter_doc(self, doc):
        """Reconstruye el texto solo con oraciones válidas."""
        valid = []
        for sent in doc.sents:
            tokens = [t for t in sent if not t.is_punct and not t.is_space]
            content = [t for t in tokens if normalize(t.text) not in STOPWORDS]
            if len(tokens) < 5 or len(content) < 2:
                continue
            first = normalize(sent[0].text)
            if first in {"este","esta","estos","estas","el","la","ello",
                         "ellos","ellas","aqui","alli"}:
                continue
            valid.append(sent.text.strip())
        return self.nlp(" ".join(valid))

    def generate(self, text: str) -> dict:
        text = text[:10000]
        doc = self._filter_doc(self.nlp(text))
        pool = _build_pool(doc)

        # Obtener todas las preguntas de cada tipo
        per_qs   = _all_per_questions(doc, pool)
        loc_qs   = _all_loc_questions(doc, pool)
        date_qs  = _all_date_questions(doc, pool)
        def_qs   = _all_def_questions(doc, pool)

        # Construir opciones para cada pregunta
        def build_question(q: dict) -> dict | None:
            if q["label"] == "DEF":
                distractors = q.get("distractores_custom", [])
            else:
                distractors = _distractors(q["respuesta"], q["label"], pool)
            if len(distractors) < 3:
                return None
            options = distractors[:3] + [q["respuesta"]]
            random.shuffle(options)
            return {
                "tipo": q["tipo"],
                "pregunta": q["pregunta"],
                "opciones": options,
                "respuesta_correcta": q["respuesta"],
            }

        buckets = {
            "fecha":      [build_question(q) for q in date_qs],
            "lugar":      [build_question(q) for q in loc_qs],
            "persona":    [build_question(q) for q in per_qs],
            "definicion": [build_question(q) for q in def_qs],
        }
        # Limpiar Nones
        buckets = {k: [q for q in v if q] for k, v in buckets.items()}

        # Intercalar en ciclo: fecha → lugar → persona → definición → fecha → ...
        cycle_order = ["fecha", "lugar", "persona", "definicion"]
        output = []
        indices = {k: 0 for k in cycle_order}

        while True:
            added = False
            for tipo in cycle_order:
                idx = indices[tipo]
                bucket = buckets[tipo]
                if idx < len(bucket):
                    output.append(bucket[idx])
                    indices[tipo] += 1
                    added = True
            if not added:
                break

        return {"preguntas": output, "total": len(output)}

# ── Demo ──────────────────────────────────────

if __name__ == "__main__":
    texto = """
    La Revolución Francesa comenzó en París en 1789 y transformó la sociedad europea.
    Napoleón Bonaparte fue un militar y estadista francés que dominó Europa a principios del siglo XIX.
    La batalla de Waterloo tuvo lugar en Bélgica en 1815 y supuso la derrota definitiva de Napoleón.
    El liberalismo es una corriente política que defiende las libertades individuales y la igualdad ante la ley.
    Marie Curie nació en Varsovia en 1867 y fue la primera mujer en ganar el Premio Nobel.
    La Torre Eiffel fue construida en París para la Exposición Universal de 1889.
    El romanticismo es un movimiento artístico que surgió en Europa a finales del siglo XVIII.
    Charles Darwin publicó El origen de las especies en Londres en 1859.
    """
    gen = FinalExamGenerator()
    result = gen.generate(texto)
    print(json.dumps(result, ensure_ascii=False, indent=2))
