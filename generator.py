"""
FinalExamGenerator v3 — sin APIs externas
==========================================
Estrategias:
  1. Detección de patrones sintácticos (definición, causal, temporal, relacional)
     → preguntas naturales según el tipo de oración
  2. Distractores coherentes via TF-IDF del dominio + restricción semántica
  3. Preguntas de comprensión global via TF-IDF vectorizer (centralidad léxica)
"""

import re
import json
import random
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ──────────────────────────────────────────────
# Stopwords — via spaCy es_core_news_sm (~570 palabras)
# Se inicializan al cargar el módulo para no repetir la carga
# ──────────────────────────────────────────────

def _load_spacy_stopwords() -> set:
    """Devuelve el set de stopwords en español de spaCy, normalizadas sin tildes."""
    import spacy.lang.es
    raw = spacy.lang.es.STOP_WORDS          # set[str] con tildes
    nfkd = unicodedata.normalize            # alias rápido
    result = set()
    for w in raw:
        norm = "".join(
            c for c in nfkd("NFKD", w.lower()) if not unicodedata.combining(c)
        )
        result.add(norm)
    return result


def normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# Cargadas una sola vez al importar el módulo
STOPWORDS: set = _load_spacy_stopwords()


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if normalize(t) not in STOPWORDS and len(t) > 2]


# ──────────────────────────────────────────────
# Patrones de oraciones
# ──────────────────────────────────────────────

RE_DEFINICION = re.compile(
    r"^(?P<sujeto>.+?)\s+(?:es|son|fue|era|son conocidos? como|se define como|"
    r"se denomina|se llama|se considera)\s+(?P<predicado>.+)$",
    re.IGNORECASE,
)

RE_CAUSAL = re.compile(
    r"(?P<efecto>.+?)\s+(?:porque|ya que|debido a|dado que|puesto que|"
    r"a causa de|gracias a|por ello|por lo que|lo que|lo cual)\s+(?P<causa>.+)",
    re.IGNORECASE,
)

RE_TEMPORAL = re.compile(
    r"\b(en\s+)?(?P<fecha>(19|20)\d{2}|siglo\s+[IVXLC]+|decada de(?: los)?\s+\d{4}s?)\b",
    re.IGNORECASE,
)

RE_RELACIONAL = re.compile(
    r"^(?P<agente>.+?)\s+(?:desarrollo|creo|fundo|invento|diseno|propuso|"
    r"escribio|publico|lanzo|presento|introdujo|establecio|organizo|"
    r"descubrio|demostro|implemento|gano|dirigio)\s+(?P<objeto>.+)$",
    re.IGNORECASE,
)


class SentencePattern:
    @staticmethod
    def detect(text: str) -> dict:
        text_norm = normalize(text.strip().rstrip("."))
        text_orig = text.strip().rstrip(".")

        # 1. Temporal
        m = RE_TEMPORAL.search(text_norm)
        if m:
            # Recuperar fecha en texto original
            m_orig = RE_TEMPORAL.search(text_orig)
            fecha = m_orig.group("fecha") if m_orig else m.group("fecha")
            contexto = RE_TEMPORAL.sub("_____", text_orig, count=1)
            return {
                "tipo": "temporal",
                "pregunta": f"¿En qué año o período ocurrió esto?\n→ {contexto}",
                "respuesta": fecha,
            }

        # 2. Definición
        m = RE_DEFINICION.match(text_norm)
        if m:
            # Extraer del texto original usando las mismas posiciones
            m_orig = RE_DEFINICION.match(text_orig)
            sujeto = (m_orig.group("sujeto") if m_orig else m.group("sujeto")).strip()
            predicado = (m_orig.group("predicado") if m_orig else m.group("predicado")).strip()
            return {
                "tipo": "definicion",
                "pregunta": f"¿Cómo se define o describe «{sujeto}»?",
                "respuesta": predicado,
            }

        # 3. Causal
        m = RE_CAUSAL.search(text_norm)
        if m:
            m_orig = RE_CAUSAL.search(text_orig)
            efecto = (m_orig.group("efecto") if m_orig else m.group("efecto")).strip()
            causa = (m_orig.group("causa") if m_orig else m.group("causa")).strip()
            return {
                "tipo": "causal",
                "pregunta": f"¿Por qué o para qué ocurrió lo siguiente?\n→ {efecto}",
                "respuesta": causa,
            }

        # 4. Relacional
        m = RE_RELACIONAL.match(text_norm)
        if m:
            m_orig = RE_RELACIONAL.match(text_orig)
            agente = (m_orig.group("agente") if m_orig else m.group("agente")).strip()
            objeto = (m_orig.group("objeto") if m_orig else m.group("objeto")).strip()
            if random.random() > 0.5:
                return {
                    "tipo": "relacional_agente",
                    "pregunta": f"¿Quién desarrolló, creó o propuso «{objeto}»?",
                    "respuesta": agente,
                }
            else:
                return {
                    "tipo": "relacional_objeto",
                    "pregunta": f"¿Qué desarrolló, creó o propuso «{agente}»?",
                    "respuesta": objeto,
                }

        return None


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

    # ── TF-IDF pool ──────────────────────────────

    def _build_tfidf_pool(self, sentences: list) -> dict:
        """
        Construye el pool de candidatos exclusivamente a partir del texto:
        - DATE  : todos los años y expresiones temporales del texto
        - PER   : personas detectadas por spaCy
        - ORG   : organizaciones detectadas por spaCy
        - LOC   : lugares detectados por spaCy (GPE + LOC)
        - CONCEPT: términos clave por TF-IDF (bigramas incluidos)
        No hay listas hardcodeadas — el dominio lo pone el texto.
        """
        pool = defaultdict(set)
        full_text = " ".join(sentences)

        # ── Fechas: años + expresiones tipo "siglo XIX", "década de 1980" ──
        pool["DATE"].update(re.findall(r"\b(?:19|20)\d{2}\b", full_text))
        pool["DATE"].update(re.findall(
            r"\bsiglo\s+[IVXLC]+\b", full_text, re.IGNORECASE
        ))
        pool["DATE"].update(re.findall(
            r"\bd[eé]cada(?:\s+de(?:\s+los)?)?\s+\d{4}s?\b", full_text, re.IGNORECASE
        ))

        # ── Entidades spaCy ──────────────────────────────────────────────
        doc = self.nlp(full_text[:10000])
        label_map = {"PER": "PER", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC"}
        for ent in doc.ents:
            mapped = label_map.get(ent.label_)
            if mapped:
                pool[mapped].add(ent.text.strip())

        # ── Sustantivos propios no capturados por NER ────────────────────
        # Secuencias de tokens con mayúscula que spaCy no clasificó como entidad
        # → se añaden al pool del tipo más probable según contexto (LOC/PER/ORG)
        ent_spans = {ent.text for ent in doc.ents}
        for token in doc:
            if (
                token.is_alpha
                and token.text[0].isupper()
                and not token.is_sent_start
                and token.text not in ent_spans
                and normalize(token.text) not in STOPWORDS
                and len(token.text) > 3
            ):
                # Sin contexto suficiente para clasificar → va a CONCEPT
                pool["CONCEPT"].add(token.text)

        # ── TF-IDF: conceptos/frases clave del dominio ───────────────────
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                ngram_range=(1, 2),
                stop_words=list(STOPWORDS),
            )
            vectorizer.fit(sentences)
            terms = vectorizer.get_feature_names_out().tolist()
            for t in terms:
                if len(t) > 4:
                    pool["CONCEPT"].add(t.capitalize())
        except Exception:
            pass

        # Convertir sets a listas
        return {k: list(v) for k, v in pool.items()}

    # ── Distractores ─────────────────────────────

    def _get_distractors(self, answer: str, label: str, pool: dict, n: int = 3) -> list:
        """
        Distractores siempre del mismo tipo semántico que la respuesta.
        Fuente exclusiva: el propio texto (pool). Para DATE se generan
        años aritméticamente si el texto tiene pocos. Para el resto,
        si el texto no tiene suficientes entidades del tipo requerido,
        la pregunta se descarta (ver generate()).
        """
        answer_clean = normalize(answer)
        candidates = list(pool.get(label, []))

        # DATE: generar años cercanos sintéticamente — siempre habrá suficientes
        if label == "DATE":
            year_match = re.search(r"\d{4}", answer)
            if year_match:
                base = int(year_match.group())
                offsets = [o for o in range(-30, 31) if o != 0]
                random.shuffle(offsets)
                for o in offsets:
                    y = str(base + o)
                    if y not in candidates:
                        candidates.append(y)
                    if len(candidates) >= n + 10:
                        break

        random.shuffle(candidates)
        distractors = []

        # Primer paso: umbral estricto (ratio < 0.55)
        for cand in candidates:
            cand_clean = normalize(cand.strip())
            if cand_clean == answer_clean:
                continue
            if SequenceMatcher(None, cand_clean, answer_clean).ratio() < 0.55:
                distractors.append(cand.strip())
            if len(distractors) == n:
                break

        # Segundo paso: si faltan, umbral relajado (cualquier distinto)
        if len(distractors) < n:
            for cand in candidates:
                c = cand.strip()
                if normalize(c) != answer_clean and c not in distractors:
                    distractors.append(c)
                if len(distractors) == n:
                    break

        return distractors[:n]

    # ── Comprensión global ────────────────────────

    def _tfidf_sim_matrix(self, sentences: list):
        """Matriz de similitud coseno entre oraciones usando TF-IDF. ~KB de RAM."""
        try:
            vec = TfidfVectorizer(stop_words=list(STOPWORDS))
            tfidf = vec.fit_transform(sentences)
            return cosine_similarity(tfidf)
        except Exception:
            # Fallback: matriz de unos si hay muy pocas palabras distintas
            n = len(sentences)
            return np.ones((n, n))

    def _global_questions(self, sentences: list) -> list:
        if len(sentences) < 3:
            return []

        sim_matrix = self._tfidf_sim_matrix(sentences)
        centrality = sim_matrix.sum(axis=1)
        central_idx = int(np.argmax(centrality))

        questions = []

        # P1: idea principal (oración más conectada léxicamente con el resto)
        opts = self._build_global_options(sentences, central_idx, sim_matrix)
        questions.append({
            "tipo": "comprension_global",
            "pregunta": "¿Cuál de las siguientes oraciones resume mejor la idea principal del texto?",
            "opciones": opts,
            "respuesta_correcta": sentences[central_idx],
            "fuente": "comprension_global",
        })

        # P2: tema secundario (oración con menor similitud léxica al núcleo)
        sims_to_central = sim_matrix[central_idx].copy()
        sims_to_central[central_idx] = 999
        distant_idx = int(np.argmin(sims_to_central))
        if distant_idx != central_idx:
            opts2 = self._build_global_options(sentences, distant_idx, sim_matrix)
            questions.append({
                "tipo": "comprension_global",
                "pregunta": "¿Qué afirmación introduce un tema distinto o secundario respecto al núcleo del texto?",
                "opciones": opts2,
                "respuesta_correcta": sentences[distant_idx],
                "fuente": "comprension_global",
            })

        return questions

    def _build_global_options(self, sentences, correct_idx, sim_matrix) -> list:
        sims = sim_matrix[correct_idx].copy()
        sims[correct_idx] = 999
        # Distractores con similitud media: no demasiado parecidos ni demasiado distintos
        order = np.argsort(np.abs(sims - 0.3))
        distractors = []
        for idx in order:
            if idx == correct_idx:
                continue
            distractors.append(sentences[idx])
            if len(distractors) == 3:
                break
        options = distractors + [sentences[correct_idx]]
        random.shuffle(options)
        return options

    # ── Filtro de oraciones ───────────────────────

    def _filter_sentences(self, doc) -> list:
        result = []
        for sent in doc.sents:
            text = sent.text.strip()
            tokens = [t.text for t in sent if not t.is_punct and not t.is_space]
            content_tokens = remove_stopwords(tokens)

            if len(tokens) < 6 or len(content_tokens) < 3:
                continue

            first = normalize(sent[0].text)
            if first in {"este","esta","estos","estas","ellos","ellas",
                         "el","la","ello","aqui","alli"}:
                continue

            if re.match(
                r"^(sin embargo|por lo tanto|ademas|no obstante|en conclusion|"
                r"por ejemplo|es decir|o sea|cabe destacar|en resumen)",
                normalize(text),
            ):
                continue

            result.append(text)
        return result

    # ── Pipeline público ──────────────────────────

    def generate(self, text: str) -> dict:
        text = text[:10000]
        doc = self.nlp(text)
        sentences = self._filter_sentences(doc)

        if not sentences:
            return {"preguntas": [], "total": 0}

        pool = self._build_tfidf_pool(sentences)
        output = []

        for sent_text in sentences:
            pattern = SentencePattern.detect(sent_text)
            if not pattern:
                continue

            answer = pattern["respuesta"]
            label = _infer_label(answer, doc)
            distractors = self._get_distractors(answer, label, pool)

            if len(distractors) < 3:
                continue

            options = distractors + [answer]
            random.shuffle(options)

            output.append({
                "tipo": pattern["tipo"],
                "pregunta": pattern["pregunta"],
                "opciones": options,
                "respuesta_correcta": answer,
                "fuente": sent_text,
            })

        output += self._global_questions(sentences)
        random.shuffle(output)
        return {"preguntas": output, "total": len(output)}


# ──────────────────────────────────────────────
# Utilidad
# ──────────────────────────────────────────────

def _infer_label(answer: str, doc) -> str:
    if re.search(r"\b(19|20)\d{2}\b", answer) or re.search(
        r"\bsiglo\s+[IVXLC]+\b", answer, re.IGNORECASE
    ):
        return "DATE"
    answer_norm = normalize(answer)
    for ent in doc.ents:
        if normalize(ent.text) in answer_norm or answer_norm in normalize(ent.text):
            if ent.label_ == "PER":
                return "PER"
            if ent.label_ == "ORG":
                return "ORG"
            if ent.label_ in ("GPE", "LOC"):
                return "LOC"
    return "CONCEPT"


# ── Demo ─────────────────────────────────────────

if __name__ == "__main__":
    texto = """
    La inteligencia artificial moderna comenzó con el trabajo de Alan Turing en 1950,
    cuando publicó su famoso artículo sobre la prueba de Turing.
    John McCarthy organizó la conferencia de Dartmouth en 1956, que marcó el nacimiento
    oficial de la IA como disciplina.
    El deep learning es una rama del aprendizaje automático basada en redes neuronales profundas.
    En 2012, AlexNet ganó el concurso ImageNet porque utilizó GPUs para entrenar redes convolucionales,
    lo que demostró el potencial del deep learning para visión artificial.
    Geoffrey Hinton desarrolló técnicas fundamentales de retropropagación que permitieron
    el entrenamiento eficiente de redes neuronales profundas.
    OpenAI fundó el laboratorio en 2015 para investigar la inteligencia artificial segura y beneficiosa.
    El aprendizaje por refuerzo es un paradigma donde un agente aprende mediante interacción
    con su entorno, recibiendo recompensas o penalizaciones.
    GPT-4 fue lanzado en 2023 y supera a modelos anteriores en razonamiento y generación de texto.
    """
    gen = FinalExamGenerator()
    result = gen.generate(texto)
    print(json.dumps(result, ensure_ascii=False, indent=2))
