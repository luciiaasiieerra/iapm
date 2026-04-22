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
from sklearn.feature_extraction.text import TfidfVectorizer


# ──────────────────────────────────────────────
# Normalización y stopwords
# ──────────────────────────────────────────────

def normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _load_stopwords() -> set:
    from spacy.lang.es import STOP_WORDS
    return {normalize(w) for w in STOP_WORDS}


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
    pool = defaultdict(set)
    full = doc.text

    for m in RE_YEAR.finditer(full):
        pool["DATE"].add(m.group("year"))
    for m in RE_PERIOD.finditer(full):
        pool["DATE"].add(m.group("period"))

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
    answer_norm = normalize(answer)
    candidates = list(pool.get(label, []))

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

    if len(result) < n:
        for c in candidates:
            cn = normalize(c)
            if cn != answer_norm and c.strip() not in result:
                result.append(c.strip())
            if len(result) == n:
                break

    return result


# ──────────────────────────────────────────────
# Extracción de preguntas por tipo
# ──────────────────────────────────────────────

def _all_per_questions(doc, pool: dict) -> list:
    seen = set()
    results = []

    all_pers = list(pool.get("PER", []))
    for token in doc:
        if (token.is_alpha and token.text[0].isupper()
                and not token.is_stop and len(token.text) > 2
                and token.ent_type_ not in ("LOC", "GPE", "ORG", "DATE")):
            if token.text not in all_pers:
                all_pers.append(token.text)

    for sent in doc.sents:
        ents = [e for e in sent.ents if e.label_ == "PER"]
        if not ents:
            continue
        ent = ents[0]
        key = normalize(ent.text)
        if key in seen:
            continue
        seen.add(key)

        distractors = [p for p in all_pers if normalize(p) != key]
        if len(distractors) < 3:
            continue

        ctx = sent.text.replace(ent.text, "_____").strip()
        results.append({
            "tipo": "persona",
            "pregunta": ctx,
            "respuesta": ent.text.strip(),
            "label": "PER",
        })
    return results


def _all_loc_questions(doc, pool: dict) -> list:
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
            "pregunta": ctx,
            "respuesta": ent.text.strip(),
            "label": "LOC",
        })
    return results


def _all_date_questions(doc, pool: dict) -> list:
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
            "pregunta": ctx.strip(),
            "respuesta": fecha,
            "label": "DATE",
        })
    return results


def _all_def_questions(doc, pool: dict) -> list:
    bad_starts = {"un","una","el","la","los","las","que","cuando",
                  "donde","como","este","esta","algo","muy"}

    pairs = []
    for sent in doc.sents:
        m = RE_DEF.match(sent.text.strip())
        if not m:
            continue
        sujeto   = m.group("sujeto").strip()
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
        return []

    results = []
    for i, (sujeto, predicado, _) in enumerate(pairs):
        distractors = [p for j, (_, p, _) in enumerate(pairs) if j != i]
        random.shuffle(distractors)
        results.append({
            "tipo": "definicion",
            "pregunta": f"«{sujeto}» es _____",
            "respuesta": predicado,
            "label": "DEF",
            "distractores_custom": distractors[:3],
        })
    return results


def _all_concept_questions(doc, pool: dict) -> list:
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
    if len(sentences) < 3:
        return []

    try:
        vec = TfidfVectorizer(
            max_features=40,
            ngram_range=(1, 1),
            stop_words=list(STOPWORDS),
        )
        tfidf  = vec.fit_transform(sentences)
        terms  = vec.get_feature_names_out()
        scores = tfidf.toarray()
    except Exception:
        return []

    ent_texts = {normalize(e.text) for e in doc.ents}
    valid_terms = [
        t for t in terms
        if normalize(t) not in ent_texts
        and not re.search(r"\d", t)
        and len(t) > 4
    ]

    if len(valid_terms) < 4:
        return []

    results      = []
    seen_terms   = set()

    for sent_idx, sent in enumerate(doc.sents):
        sent_text = sent.text.strip()
        if len(sent_text) < 20:
            continue

        row = scores[sent_idx] if sent_idx < len(scores) else None
        if row is None:
            continue

        best_term  = None
        best_score = 0.0
        for t in valid_terms:
            t_idx = list(terms).index(t) if t in terms else -1
            if t_idx == -1:
                continue
            sc = row[t_idx]
            if sc > best_score and normalize(t) not in seen_terms:
                pattern = re.compile(r'' + re.escape(t) + r'', re.IGNORECASE)
                if pattern.search(sent_text):
                    best_score = sc
                    best_term  = t

        if not best_term or best_score < 0.05:
            continue

        seen_terms.add(normalize(best_term))

        pattern = re.compile(r'' + re.escape(best_term) + r'', re.IGNORECASE)
        ctx = pattern.sub("_____", sent_text, count=1).strip()

        distractors = [
            t.capitalize() for t in valid_terms
            if normalize(t) != normalize(best_term)
        ]
        random.shuffle(distractors)
        distractors = distractors[:3]

        if len(distractors) < 3:
            continue

        results.append({
            "tipo": "concepto",
            "pregunta": ctx,
            "respuesta": best_term.capitalize(),
            "label": "CONCEPT",
            "distractores_custom": distractors,
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
        valid = []
        for sent in doc.sents:
            tokens  = [t for t in sent if not t.is_punct and not t.is_space]
            content = [t for t in tokens if normalize(t.text) not in STOPWORDS]
            if len(tokens) < 5 or len(content) < 2:
                continue
            first = normalize(sent[0].text)
            if first in {"este","esta","estos","estas","el","la","ello",
                         "ellos","ellas","aqui","alli"}:
                continue
            valid.append(sent.text.strip())
        return self.nlp(" ".join(valid))

    def generate(self, text: str, existing_questions: list[str] | None = None) -> dict:
        text = text[:10000]
        doc  = self._filter_doc(self.nlp(text))
        pool = _build_pool(doc)

        per_qs  = _all_per_questions(doc, pool)
        loc_qs  = _all_loc_questions(doc, pool)
        date_qs = _all_date_questions(doc, pool)
        def_qs  = _all_def_questions(doc, pool)
        con_qs  = _all_concept_questions(doc, pool)

        def build_question(q: dict) -> dict | None:
            if q["label"] in ("DEF", "CONCEPT"):
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
            "concepto":   [build_question(q) for q in con_qs],
        }
        buckets = {k: [q for q in v if q] for k, v in buckets.items()}

        cycle_order = ["fecha", "lugar", "persona", "concepto", "definicion"]
        queues = {k: list(buckets[k]) for k in cycle_order}
        output = []

        while any(queues[k] for k in cycle_order):
            for tipo in cycle_order:
                if queues[tipo]:
                    output.append(queues[tipo].pop(0))

        # ── Filtrar preguntas ya existentes en el banco ──────────────
        if existing_questions:
            existing_normalized = {normalize(q) for q in existing_questions}
            output = [
                q for q in output
                if normalize(q["pregunta"]) not in existing_normalized
            ]
        # ────────────────────────────────────────────────────────────

        return {"preguntas": output, "total": len(output)}