import random
import spacy
import json
import re
from difflib import SequenceMatcher


class FinalExamGenerator:
    def __init__(self):
        print("Inicializando motor de precisión...")

        try:
            # 🔥 spaCy optimizado para baja RAM
            self.nlp = spacy.load(
                "es_core_news_sm",
                disable=["parser", "tagger", "lemmatizer"]
            )
        except:
            import os
            os.system("python -m spacy download es_core_news_sm")
            self.nlp = spacy.load(
                "es_core_news_sm",
                disable=["parser", "tagger", "lemmatizer"]
            )

        self.kb_fallback = {
            "PER": ["Ada Lovelace", "Claude Shannon", "Marvin Minsky", "Geoffrey Hinton"],
            "LOC": ["Silicon Valley", "Stanford", "MIT", "Londres"],
            "ORG": ["Google", "Microsoft", "DeepMind", "IBM", "Meta", "Anthropic"],
            "DATE": ["1945", "1950", "1968", "1980", "1997", "2012", "2020"],
            "CONCEPT": ["Algoritmo", "Red Neuronal", "Cómputo en la nube", "Big Data", "Criptografía"]
        }

        print("Sistema listo.")

    def clean_text(self, text):
        cleaned = re.sub(
            r'^(el|la|los|las|un|una|unos|unas|en|de|del|al)\s+',
            '',
            text,
            flags=re.IGNORECASE
        ).strip()

        return cleaned.capitalize() if not cleaned.isdigit() else cleaned

    def is_year(self, text):
        return bool(re.search(r'\b(19|20)\d{2}\b', text))

    def get_distractors(self, target, label, kb_global):
        candidates = list(kb_global.get(label, set())) + self.kb_fallback.get(label, self.kb_fallback["CONCEPT"])

        target_clean = self.clean_text(target)
        candidates = list(set([self.clean_text(c) for c in candidates]))

        final_distractors = []
        random.shuffle(candidates)

        for cand in candidates:
            ratio = SequenceMatcher(None, cand.lower(), target_clean.lower()).ratio()
            if cand.lower() != target_clean.lower() and ratio < 0.45:
                final_distractors.append(cand)

            if len(final_distractors) == 3:
                break

        # 🔥 fallback para fechas
        if label == "DATE" or self.is_year(target_clean):
            year_match = re.search(r'\d{4}', target_clean)
            year = int(year_match.group()) if year_match else 2023

            while len(final_distractors) < 3:
                new_year = str(year + random.randint(-15, 15))
                if new_year != str(year) and new_year not in final_distractors:
                    final_distractors.append(new_year)

        return final_distractors

    def generate(self, text):
        # 🔥 limitador de seguridad RAM
        text = text[:5000]

        doc = self.nlp(text)

        kb = {"PER": set(), "LOC": set(), "DATE": set(), "CONCEPT": set(), "ORG": set()}

        for ent in doc.ents:
            if ent.label_ == "PER":
                kb["PER"].add(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:
                kb["LOC"].add(ent.text)
            elif ent.label_ == "ORG":
                kb["ORG"].add(ent.text)
            elif ent.label_ == "DATE" or self.is_year(ent.text):
                kb["DATE"].add(ent.text)

        output = []

        for sent in doc.sents:

            # filtro básico
            if len(sent) == 0:
                continue

            if sent[0].lower_ in ["este", "esta", "estos", "estas", "ellos", "ellas"]:
                continue

            target, label = None, None

            # 1. fechas
            for ent in sent.ents:
                if ent.label_ == "DATE" or self.is_year(ent.text):
                    target, label = ent.text, "DATE"
                    break

            # 2. entidades
            if not target:
                for ent in sent.ents:
                    if ent.label_ in ["PER", "ORG", "GPE", "LOC"]:
                        target = ent.text
                        label = "PER" if ent.label_ == "PER" else \
                                "ORG" if ent.label_ == "ORG" else "LOC"
                        break

            # 3. fallback ligero (regex en vez de noun_chunks)
            if not target:
                matches = re.findall(
                    r'\b[A-ZÁÉÍÓÚ][a-záéíóú]+\s[A-ZÁÉÍÓÚ][a-záéíóú]+\b',
                    sent.text
                )
                if matches:
                    target, label = matches[0], "CONCEPT"

            if target and label:
                target_final = self.clean_text(target)
                distractors = self.get_distractors(target, label, kb)

                if len(distractors) >= 3:
                    options = distractors + [target_final]
                    random.shuffle(options)

                    prefixes = {
                        "PER": "¿Quién es la persona?",
                        "LOC": "¿En qué lugar?",
                        "ORG": "¿Qué organización?",
                        "DATE": "¿En qué fecha o año?",
                        "CONCEPT": "¿A qué concepto se refiere?"
                    }

                    pregunta = sent.text.replace(target, "_____").strip()

                    output.append({
                        "tipo": "test",
                        "pregunta": f"{prefixes.get(label)}: {pregunta}",
                        "opciones": options,
                        "respuesta_correcta": target_final
                    })

        return {"preguntas": output}


# --- EJECUCIÓN ---
texto_ejemplo = """
La inteligencia artificial moderna comenzó con el trabajo de Alan Turing en 1950.
John McCarthy organizó la famosa conferencia de Dartmouth en 1956.
John von Neumann desarrolló la arquitectura de computadores en la década de 1940.
La empresa OpenAI fue fundada para investigar la inteligencia artificial segura.
Actualmente, el desarrollo tecnológico ocurre en San Francisco.
El deep learning utiliza redes neuronales para procesar datos masivos.
En 2023 se lanzaron modelos de lenguaje revolucionarios.
"""

gen = FinalExamGenerator()
result = gen.generate(texto_ejemplo)

print(json.dumps(result, ensure_ascii=False, indent=2))